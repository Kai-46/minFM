# minFM: Minimal Flow Matching

A clean, modular, and scalable training system for training text-to-{image,video} Flow Matching (FM) models.

![](./resources/teasor.png)

*First two rows: image generation results using the trained flux-tiny model via our minFM training system on ImageNet 256. Third row: image generation results from inference using a loaded checkpoint from [FLUX.1 [dev]](https://huggingface.co/black-forest-labs/FLUX.1-dev)*

## Requirements

- NVIDIA GPUs
- Linux environment

**GPU Configuration Notes:**
- You must ensure the total number of GPUs is divisible by the `shard_size` parameter in your config
- The total number of available GPUs must be divisible by the number of GPUs in each DIT balancer sharding group, which is specified by `dit_balancer_specs` (e.g., "g1n4" means 4 GPUs per group, so total GPUs must be divisible by 4)
- See the Configuration section below for details on adjusting these parameters

## Quick Start

Before running training or inference, prepare the required data and checkpoints.

```bash
# Before running the script, specify the folders for model caching (e.g., VAE) and data:
export MINFM_CACHE_DIR="<minfm-cache-dir>"
export MINFM_DATA_DIR="<minfm-data-dir>"

# Prepare: cache everything required for inference and training
# Requires HF_TOKEN (https://huggingface.co/settings/tokens)
# This downloads:
#  - T5, CLIP, FLUX VAE, and FLUX 1.dev checkpoints to MINFM_CACHE_DIR
#  - ImageNet dataset to MINFM_DATA_DIR
# Requires ~210G of storage in total.
export HF_TOKEN="<your-hf-token>"
bash ./scripts/cache_everything.sh

# Run below to precompute the VAE latents.
torchrun --nproc_per_node=8 scripts/vae_precompute.py --config configs/flux_tiny_imagenet.yaml --output_dir $MINFM_DATA_DIR/imagenet/vae_latents
```

For training, the easiest way to get started is using the provided training script:

```bash
# Run training with the default config
bash run.sh train ./configs/flux_tiny_imagenet.yaml

# Run training with a custom config
bash run.sh train path/to/your/config.yaml

# To use Weights & Biases logging, set wandb_mode to "online" in the config
# and then set the WANDB_API_KEY environment variable
WANDB_API_KEY=<YOUR_WANDB_KEY> bash run.sh train path/to/your/config.yaml
```

For inference (text-to-image generation):

```bash
# Run inference using a pretrained FLUX model
# See the `inferencer` section for inference parameters
# Note that the shard_size and dit_balancer_specs in the config are pre-set
#    for 4*K GPUs; adjust the values to accommodate your available GPUs. 
bash run.sh inference ./configs/flux_inference.yaml

# Run inference with custom config
bash run.sh inference path/to/your/config.yaml
```

The `run.sh` script automatically:
- ✅ Sets up the environment with `uv`
- ✅ Runs distributed training/inference with proper settings
- ✅ Starts background http server for easy inspection of intermediate results
- ✅ Starts background periodic checkpoint clean-up

## ImageNet Training Results

We provide a complete training example using the tiny FLUX DiT model (560.25M parameters) trained on ImageNet with the [`flux_tiny_imagenet.yaml`](configs/flux_tiny_imagenet.yaml) configuration.

### 📊 Training Resources
- **Training Time**: ~4 days on 8× H100 GPUs  
- **Total Steps**: 380K steps with 1K batch size
- **Model Size**: 560.25M parameters

### 📈 Available Training Artifacts
All training artifacts are hosted on the [HuggingFace minFM repository](https://huggingface.co/datasets/Kai-46/minFM/tree/main):

- **[📈 Training & Validation Curves](https://huggingface.co/datasets/Kai-46/minFM/blob/main/flux-tiny_wandb_imagenet_training_run.pdf)** - Complete W&B training metrics and loss curves

- **[🎨 Intermediate Visualizations](https://huggingface.co/datasets/Kai-46/minFM/blob/main/flux-tiny_imagenet_intermediate_results.tar.gz)** - Generated samples every 2k steps
  ```bash
  # Download, extract, and view locally
  wget https://huggingface.co/datasets/Kai-46/minFM/resolve/main/flux-tiny_imagenet_intermediate_results.tar.gz
  tar -xzf flux-tiny_imagenet_intermediate_results.tar.gz
  cd flux-tiny_imagenet_intermediate_results
  python -m http.server 8000
  # Open http://localhost:8000 in your browser
  ```

- **[💾 Final Checkpoint](https://huggingface.co/datasets/Kai-46/minFM/resolve/main/flux-tiny_imagenet_step_00380000.tar.gz)** - Ready-to-use model weights at step 380k; this contains float32 model, ema, optimizer.

- **Evaluation Metrics**

  Sampling solver: [50-step DDIM-style SDE](utils_fm/sampler.py)

  | CFG Scale | Inception Score | FID   | sFID  | Precision | Recall |
  |-----------|----------------|-------|-------|-----------|--------|
  | 1.5       | 248.19         | 3.44  | 9.48  | 0.818     | 0.515  |
  | 2.0       | 342.52         | 5.33  | 8.61  | 0.884     | 0.435  |
  | 5.0       | 478.90         | 16.54 | 10.08 | 0.934     | 0.205  |

  Reproduction:
  ```
  # Download the above checkpoint to ./experiments/flux_tiny_imagenet
  # Comment out the inferencer section in ./configs/flux_tiny_imagenet.yaml intended for metrics computation

  # Sample images
  bash run.sh inference ./configs/flux_tiny_imagenet.yaml

  # Compute metrics
  python scripts/eval_imagenet_metrics.py <path/to/sampled/images.npz>
  ```

## Manual Setup

If you prefer manual setup:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --no-dev

# For development:
uv sync; pre-commit install

# Run training/inference (adjust --nproc_per_node to match your GPU count)
uv run torchrun --nproc_per_node=<num_gpus> -m entrypoint --config path/to/your/config.yaml --mode <train/inference>
```

## Configuration

The system uses YAML configuration files. See `configs/flux_tiny_imagenet.yaml` for a complete example.

### GPU-Specific Configuration

When using a different number of GPUs than the default configs, you need to adjust two key parameters:

1. **`shard_size`**: Found in the FSDP sections of model components (text_encoder, clip_encoder, denoiser)
   - Must be a divisor of your total GPU count
   - Example: For 6 GPUs, you can use `shard_size: 1`, `2`, `3`, or `6`

2. **`dit_balancer_specs`**: Found in the balancer section
   - Format: "1x{gpus_per_group}" where {gpus_per_group} specifies GPUs per sharding group
   - Your total GPU count must be divisible by this number
   - Example: "1x4" for 4 GPUs per group (works with 4, 8, 12, 16... total GPUs)

**Example for 4 GPUs:**
```yaml
model:
  text_encoder:
    fsdp:
      shard_size: 1  # or 2, or 4
  denoiser:
    fsdp:
      shard_size: 1  # or 2, or 4
  balancer:
    dit_balancer_specs: "g1n4"  # 4 GPUs per group, total GPUs (4) divisible by 4
```

**Example for 8 GPUs with 4 GPUs per group:**
```yaml
model:
  text_encoder:
    fsdp:
      shard_size: 1  # or 2, 4, or 8
  denoiser:
    fsdp:
      shard_size: 1  # or 2, 4, or 8
  balancer:
    dit_balancer_specs: "g1n8"  # 8 GPUs per group, total GPUs (8) divisible by 8
                               # you can also try 1x4, which means 2 4-GPU groups
```

### Key Components:
- **Denoiser**: Primary model architecture
- **VAE**: Image compression/decompression
- **Text Encoder (T5)**: Text embeddings
- **Text Encoder (CLIP)**: Text embeddings 
- **Patchifier**: Image tokenization into patches
- **TimeSampler**: Timestep sampling
- **TimeWarper**: Adaptive timestep scheduling based on sequence length
- **TimeWeighter**: Loss weighting based on timesteps

## Features

- **🔄 Native Packed Sequences** - Operate natively on packed interleaved text and image sequences
- **⚖️ KnapFormer Sequence Balancer** - Balance compute workloads across GPUs for optimal performance
- **🔄 Configurable Gradient Accumulation** - Automatic gradient accumulation with configurable total batch sizes
- **💾 Flexible Checkpoint Loading** - Selective loading of model, EMA, optimizer, scheduler and step components
- **🚀 Distributed Training and Async Checkpointing** - FSDP2 support with EMA
- **📦 Modular Design** - Mix and match components with structured YAML files
- **⚡ Highly Optimized** - FlashAttention variable-length support for H100/A100

## Miscellaneous
```
# Download Parti prompts
curl -s https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv \
    | tail -n +2 | cut -f1 | awk 'NF' \
    > resources/parti_prompts.txt
```

## License

This project is licensed under the Apache-2.0 License — see the [LICENSE](LICENSE) file for details.

## Citations
```bash
# If you use this repo, please cite:
@misc{zhang2025minfm,
  title={minFM},
  author={Kai, Zhang and Peng, Wang and Sai, Bi and Jianming, Zhang and Yuanjun, Xiong},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/Kai-46/minFM/}},
  year={2025}
}

# If you use the KnapFormer sequence balancer, please also cite:
@misc{zhang2025knapformer,
  title={KnapFormer},
  author={Kai, Zhang and Peng, Wang and Sai, Bi and Jianming, Zhang and Yuanjun, Xiong},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/Kai-46/KnapFormer/}},
  year={2025}
}

# If you use the energy-preserving cfg in utils/sampler.py, please also cite:
@article{zhang2024ep,
  title={EP-CFG: Energy-Preserving Classifier-Free Guidance},
  author={Zhang, Kai and Luan, Fujun and Bi, Sai and Zhang, Jianming},
  journal={arXiv preprint arXiv:2412.09966},
  year={2024}
}
```

## Notes
This repository may be relocated to the [adobe-research organization](https://github.com/adobe-research), with this copy serving as a mirror.
