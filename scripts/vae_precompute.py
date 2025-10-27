#!/usr/bin/env python3
"""
VAE Precompute Script

This script uses the ImagenetDataModule to load images from the ImageNet dataset
and computes VAE latent embeddings for each image, saving them as individual .pt files.

Usage:
    # Single GPU (with bfloat16 by default):
    python scripts/vae_precompute.py --config configs/flux_tiny_imagenet.yaml --output_dir /path/to/output

    # Single GPU (disable bfloat16):
    python scripts/vae_precompute.py --config configs/flux_tiny_imagenet.yaml --output_dir /path/to/output --no-bfloat16

    # Multi-GPU distributed processing:
    torchrun --nproc_per_node=4 scripts/vae_precompute.py --config configs/flux_tiny_imagenet.yaml --output_dir /path/to/output

    # Archive output to tar.gz:
    python scripts/vae_precompute.py --config configs/flux_tiny_imagenet.yaml --output_dir /path/to/output --archive-output

Note: In distributed mode, each rank processes a subset of the data. The dataset is automatically 
partitioned across all available processes, with each process handling different images.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import yaml

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from data.imagenet import ImagenetDataModule, ImagenetDataModuleParams
from models.flux_vae import AutoEncoder, AutoEncoderParams
from trainers import load_config
from utils.log import get_logger

logger = get_logger(__name__)


def setup_distributed():
    """Initialize distributed processing."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize the process group
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single process mode
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed processing."""
    if dist.is_initialized():
        dist.destroy_process_group()


def expand_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively expand environment variables in config."""
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith('$'):
        # Handle environment variable expansion
        env_var = config[1:]  # Remove $
        return os.environ.get(env_var, config)
    else:
        return config


def create_vae_model(vae_config: dict[str, Any], device: torch.device, use_bfloat16: bool = True, rank: int = 0) -> AutoEncoder:
    """Create and load the VAE model."""
    if rank == 0:
        logger.info("Loading VAE model...")
    
    # Create VAE parameters
    vae_params = AutoEncoderParams(**vae_config.get('params', {}))
    
    # Create the model
    vae = AutoEncoder(vae_params, from_pretrained=True)
    vae.to(device)
    
    # Convert to bfloat16 if requested
    if use_bfloat16:
        vae = vae.to(torch.bfloat16)
        if rank == 0:
            logger.info("VAE model converted to bfloat16")
    
    vae.eval()
    
    if rank == 0:
        dtype_info = "bfloat16" if use_bfloat16 else "float32"
        logger.info(f"VAE model loaded with {sum(p.numel() for p in vae.parameters())} parameters ({dtype_info})")
    return vae


def create_data_module(data_config: dict[str, Any], rank: int = 0) -> ImagenetDataModule:
    """Create the ImageNet data module."""
    if rank == 0:
        logger.info("Setting up ImageNet data module...")
    
    # Expand environment variables in data config
    data_config = expand_env_vars(data_config)
    
    # Create data parameters
    data_params = ImagenetDataModuleParams(**data_config.get('params', {}))
    
    # Create data module
    data_module = ImagenetDataModule(data_params)
    
    if rank == 0:
        logger.info(f"Data module created with train: {len(data_module.datasets['train'])} images, "
                   f"val: {len(data_module.datasets['val'])} images")
    
    return data_module


def get_image_filename(data_meta: dict, idx: int, split: str) -> str:
    """Get a unique filename for the latent file based on image metadata."""
    if 'img_path' in data_meta:
        # Use the original image filename
        img_path = Path(data_meta['img_path'])
        return f"{img_path.stem}.pt"
    else:
        # Fallback to index-based naming
        return f"{split}_{idx:08d}.pt"


def encode_and_save_dataset(
    vae: AutoEncoder,
    dataset: Any,
    output_dir: Path,
    split: str,
    batch_size: int = 8,
    device: torch.device = torch.device('cuda'),
    use_bfloat16: bool = True,
    rank: int = 0,
    world_size: int = 1
) -> None:
    """Encode a dataset and save latents as individual .pt files."""
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wait for rank 0 to create directories
    if dist.is_initialized():
        dist.barrier()
    
    # Create distributed sampler if using multiple processes
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    
    # Create dataloader with distributed sampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    # Only log on rank 0 to avoid spam
    if rank == 0:
        logger.info(f"Starting to encode {split} split ({len(dataset)} images) across {world_size} processes...")
    
    # Calculate the starting index for this rank
    if sampler is not None:
        current_idx = rank * len(sampler)
    else:
        current_idx = 0
    
    # Disable tqdm on non-zero ranks to avoid progress bar spam
    disable_tqdm = (rank != 0)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Encoding {split} (rank {rank})", disable=disable_tqdm)):
            images = batch['raw_images'].to(device)  # Shape: (B, 3, H, W)
            
            # Convert to bfloat16 if enabled
            if use_bfloat16:
                images = images.to(torch.bfloat16)
            
            # Encode images to latents
            try:
                latents = vae.encode(images)  # Shape: (B, latent_channels, H//8, W//8)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM error on batch {batch_idx}. Try reducing batch_size.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            # Save each latent individually
            for i in range(latents.shape[0]):
                # Calculate the global dataset index for this sample
                if sampler is not None:
                    # In distributed mode, calculate the global index
                    # Each rank processes different indices based on DistributedSampler logic
                    local_idx = batch_idx * batch_size + i
                    global_idx = rank + local_idx * world_size
                else:
                    global_idx = current_idx + i
                
                # Get filename
                file_stem = batch['file_stems'][i]
                filename = f"{file_stem}.pt"

                latent_path = split_output_dir / filename
                
                # Save latent in the same precision as inference
                latent_to_save = latents[i].cpu()
                if use_bfloat16:
                    latent_to_save = latent_to_save.to(torch.bfloat16)
                
                torch.save(latent_to_save, latent_path)
            
            current_idx += latents.shape[0]

            # Clear cache periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
    
    # Synchronize before final logging
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        logger.info(f"Completed encoding {split} split. All ranks saved latent files to {split_output_dir}")
    elif world_size > 1:
        logger.info(f"Rank {rank}: Processed {current_idx} samples for {split} split")


def main():
    parser = argparse.ArgumentParser(description="Precompute VAE latents for ImageNet dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the precomputed latents"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for encoding (default: 8)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["val", "train"],
        help="Dataset splits to process (default: train val)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (default: cuda)"
    )
    parser.add_argument(
        "--archive-output",
        action="store_true",
        help="Archive the output directory to a tar.gz file. The file will be named <output_dir>.tar.gz"
    )
    parser.add_argument(
        "--use-bfloat16",
        action="store_true",
        default=True,
        help="Use bfloat16 for inference and storage to save memory and disk space (default: True)"
    )
    parser.add_argument(
        "--no-bfloat16",
        dest="use_bfloat16",
        action="store_false",
        help="Disable bfloat16 and use float32 instead"
    )
    
    args = parser.parse_args()
    
    # Initialize distributed processing
    rank, world_size, local_rank = setup_distributed()
    
    # Setup device - use local_rank for CUDA device selection in distributed mode
    if world_size > 1 and args.device.startswith('cuda'):
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device)
    
    # Create output directory (only on rank 0)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wait for rank 0 to create output directory
    if dist.is_initialized():
        dist.barrier()
    
    # Only log on rank 0 to avoid spam
    if rank == 0:
        logger.info(f"Starting VAE precomputation with {world_size} process(es)")
        logger.info(f"Config: {args.config}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Splits to process: {args.splits}")
        logger.info(f"Use bfloat16: {args.use_bfloat16}")
    
    logger.info(f"Rank {rank}/{world_size} initialized on device {device}")
    
    # Load configuration
    if rank == 0:
        logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Create VAE model
    vae_config = config['model']['vae']
    vae = create_vae_model(vae_config, device, args.use_bfloat16, rank)
    
    # Create data module
    data_config = config['data']
    data_config['params']['use_precomputed_latents'] = False # We need to load the images to compute the latents
    data_module = create_data_module(data_config, rank)

    # Process each split
    for split in args.splits:
        if split not in data_module.datasets:
            if rank == 0:
                logger.warning(f"Split '{split}' not found in datasets. Available splits: {list(data_module.datasets.keys())}")
            continue
            
        dataset = data_module.datasets[split]
        encode_and_save_dataset(
            vae=vae,
            dataset=dataset,
            output_dir=output_dir,
            split=split,
            batch_size=args.batch_size,
            device=device,
            use_bfloat16=args.use_bfloat16,
            rank=rank,
            world_size=world_size
        )
    
    # Synchronize all processes before completion
    if dist.is_initialized():
        dist.barrier()
    
    if rank == 0:
        logger.info("VAE precomputation completed successfully!")
    
    if args.archive_output and rank == 0:
        logger.info("Archiving output directory...")
        os.system(f"tar -czf {output_dir}.tar.gz -C {output_dir.parent} {output_dir.name} > /dev/null 2>&1")
        logger.info(f"Output directory archived to {output_dir}.tar.gz")

    # Cleanup distributed processing
    cleanup_distributed()


if __name__ == "__main__":
    main()
