from dataclasses import dataclass, fields
import itertools
import json
import os
from typing import Any, Literal, cast

from knapformer import SequenceBalancer
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from models.patchifier import Patchifier
from utils.fsdp import fwd_only_mode
from utils.log import get_logger, human_readable_number
from utils.pack import pack_reduce
from utils.vis import generate_html_gallery
from utils_fm.noiser import NoiserProtocol
from utils_fm.sampler import FlowSampler, energy_preserve_cfg

logger = get_logger(__name__)


@dataclass
class FMDataContext:
    """
    Flow Matching data context for managing (txt, img) pairs during training.
    All fields are optional to support progressive data flow where each field
    is filled in by different operators (frozen and trainable).

    Attributes:
        raw_texts: List of text prompts as strings
        raw_images: Raw image tensors (b, c, h, w)
        raw_latents: Raw latent tensors (b, c, f, h // p, w // p)
        txt: Text embeddings (l1+l2+...+ln, d_txt)
        txt_datum_lens: Length of each text sequence (n,)
        txt_position_ids: Position IDs for text tokens (l1+l2+...+ln, d_position)
        img_clean: Clean image patches/latents (l1+l2+...+ln, d_img)
        img: Noised image patches/latents (l1+l2+...+ln, d_img)
        img_datum_lens: Length of each image sequence (n,)
        img_position_ids: Position IDs for image patches (l1+l2+...+ln, d_position)
        timesteps: Diffusion timesteps (n,)
        vec: Guidance vector for conditioning (n, d_vec)
    """

    """Filled by Dataloader"""
    raw_texts: list[str] | None = None
    raw_images: torch.Tensor | None = None  # (n, c, f, h, w)
    raw_latents: torch.Tensor | None = None  # (n, c, f, h // p, w // p)

    """Filled by FrozenOps"""
    """Frozen T5 Ops"""
    txt: torch.Tensor | None = None  # (l1+l2+...+ln, d_txt)
    txt_datum_lens: torch.Tensor | None = None  # (n,)
    txt_position_ids: torch.Tensor | None = None  # (l1+l2+...+ln, d_position)
    """Frozen CLIP Ops"""
    vec: torch.Tensor | None = None  # (n, d_vec)
    """Frozen VAE Ops and Patchifier Ops"""
    img_clean: torch.Tensor | None = None  # (l1+l2+...+ln, d_img)
    img_datum_lens: torch.Tensor | None = None  # (n,)
    img_position_ids: torch.Tensor | None = None  # (l1+l2+...+ln, d_position)
    """Frozen Flow Noiser Ops"""
    img: torch.Tensor | None = None  # (l1+l2+...+ln, d_img)
    timesteps: torch.Tensor | None = None  # (n,)
    timestep_weights: torch.Tensor | None = None  # (n,)
    img_v_truth: torch.Tensor | None = None  # (l1+l2+...+ln, d_img)
    """Filled by TrainableOps"""
    img_v_pred: torch.Tensor | None = None  # (l1+l2+...+ln, d_img)
    loss_vec: torch.Tensor | None = None  # (n,)
    loss: torch.Tensor | None = None  # (1,)
    """Stats"""
    num_tokens: int | None = None

    def summarize(self) -> str:
        """Return a human-readable table summarizing tensor attributes.

        For every tensor attribute that is not ``None`` this method lists:
        1. Shape;
        2. Mean;
        3. Standard deviation;
        4. Minimum value;
        5. Maximum value.
        6. Dtype.

        The summary string is also logged for quick inspection.
        """

        headers = ("Tensor", "Shape", "Mean", "Std", "Min", "Max", "Dtype")
        line_sep = "-" * 105
        summary_lines: list[str] = [
            f"{headers[0]:20} | {headers[1]:30} | "
            f"{headers[2]:>12} | {headers[3]:>12} | {headers[4]:>12} | {headers[5]:>12} | "
            f"{headers[6]:>12}",
            line_sep,
        ]

        # Iterate over dataclass fields and collect stats for tensor attributes
        for field in fields(self):
            name = field.name
            value = getattr(self, name)

            # Skip non-tensor values that are not None (e.g., raw_texts list)
            if value is None or not torch.is_tensor(value):
                continue

            tensor = value
            # If the tensor is a DTensor, operate on its local shard for stats
            if isinstance(tensor, torch.distributed.tensor.DTensor):
                tensor = tensor._local_tensor

            shape_str = "x".join(map(str, tensor.shape))
            tensor_f = tensor.to(torch.float32)
            mean_v = tensor_f.mean().item()
            std_v = tensor_f.std().item()
            min_v = tensor_f.min().item()
            max_v = tensor_f.max().item()
            dtype_str = str(tensor.dtype).replace("torch.", "")

            summary_lines.append(
                f"{name:20} | {shape_str:30} | {mean_v:12.4f} | {std_v:12.4f} | {min_v:12.4f} | {max_v:12.4f} | {dtype_str:12}"
            )

        summary_str = "\n".join(summary_lines)
        logger.info("\n" + summary_str)
        return summary_str


@dataclass
class LatentFM:
    """
    Dataclass for managing all components in a latent flow matching model.

    Attributes:
        text_encoder: Text encoder for processing text prompts (any nn.Module implementation)
        clip_encoder: CLIP encoder for visual-textual understanding (any nn.Module implementation)
        vae: Variational autoencoder for encoding/decoding images
        patchifier: Patchifier for converting between images and patches
        denoiser: Main denoiser transformer model (supports any nn.Module implementation)
        ema_denoiser: EMA (Exponential Moving Average) version of the denoiser for inference
        flow_noiser: FlowNoiser for applying noise to clean data during training
        time_sampler: TimeSampler for sampling timesteps
        time_warper: TimeWarper for adaptive timestep scheduling based on sequence length
        time_weighter: TimeWeighter for loss weighting based on timestep distribution
        dit_balancer: SequenceBalancer for managing sequence lengths in batch processing
    """

    text_encoder: nn.Module | None = None
    clip_encoder: nn.Module | None = None
    vae: nn.Module | None = None
    patchifier: Patchifier | None = None
    denoiser: nn.Module | None = None
    ema_denoiser: nn.Module | None = None
    flow_noiser: nn.Module | None = None
    time_sampler: nn.Module | None = None
    time_warper: nn.Module | None = None
    time_weighter: nn.Module | None = None
    dit_balancer: SequenceBalancer | None = None

    def summarize(self) -> str:
        """Return a human-readable table of parameter counts for the main sub-modules.

        For each available component (``text_encoder``, ``clip_encoder``, ``vae``,
        ``denoiser``, ``ema_denoiser``) the function lists:
        1. total parameters
        2. trainable parameters (``requires_grad=True``)
        3. frozen parameters (``requires_grad=False``)

        The summary is returned as a string and also printed to stdout so that
        users can quickly inspect it.
        """

        def _count_params(module: nn.Module) -> tuple[int, int, int, int, str, str]:
            """Return (global_total, global_trainable, local_total, local_trainable, device, dtype).

            *global_* counts refer to the full parameter sizes (``p.numel()``).
            *local_* counts consider only the local shards when ``p`` is a
            ``DTensor`` produced by FSDP2; otherwise they equal the global counts.
            """
            first_param = next(module.parameters())
            device_str = first_param.device.type
            dtype_str = str(first_param.dtype).replace("torch.", "")

            global_total = 0
            global_trainable = 0
            local_total = 0
            local_trainable = 0

            for p in module.parameters():
                # Global counts (always available)
                numel_global = p.numel()
                global_total += numel_global
                if p.requires_grad:
                    global_trainable += numel_global

                # Local counts (handle DTensor)
                if isinstance(p, torch.distributed.tensor.DTensor):
                    local_view = p._local_tensor  # Tensor representing the local shard
                    numel_local = local_view.numel()
                else:
                    numel_local = numel_global
                local_total += numel_local
                if p.requires_grad:
                    local_trainable += numel_local

            return global_total, global_trainable, local_total, local_trainable, device_str, dtype_str

        headers = ("Module", "Global", "Trainable", "Frozen", "Local", "Device", "Dtype")
        line_sep = "-" * 105
        summary_lines: list[str] = [
            f"{headers[0]:20} | {headers[1]:>12} | "
            f"{headers[2]:>12} | {headers[3]:>12} | "
            f"{headers[4]:>12} | {headers[5]:>12} | "
            f"{headers[6]:>12}",
            line_sep,
        ]

        modules_to_check: list[tuple[str, nn.Module | None]] = [
            ("text_encoder", self.text_encoder),
            ("clip_encoder", self.clip_encoder),
            ("vae", self.vae),
            ("denoiser", self.denoiser),
        ]
        if self.ema_denoiser is not None:
            modules_to_check.append(("ema_denoiser", self.ema_denoiser))

        for name, module in modules_to_check:
            if module is None:
                continue
            g_total, g_train, l_total, _, device_str, dtype_str = _count_params(module)
            g_frozen = g_total - g_train

            summary_lines.append(
                f"{name:20} | "
                f"{human_readable_number(g_total):>12} | "
                f"{human_readable_number(g_train):>12} | "
                f"{human_readable_number(g_frozen):>12} | "
                f"{human_readable_number(l_total):>12} | "
                f"{device_str:>12} | "
                f"{dtype_str:>12}"
            )

        summary_str = "\n".join(summary_lines)

        # Print for convenience
        logger.info("\n" + summary_str)
        return summary_str


@dataclass
class FrozenOps:
    lfm: LatentFM

    @torch.no_grad()
    def __call__(self, data_batch: FMDataContext, txt_drop_prob: float = 0.1) -> FMDataContext:
        """Drop raw text with a certain probability in i.i.d. manner (in-place)."""
        assert data_batch.raw_texts is not None, "raw_texts must be provided"
        assert self.lfm.patchifier is not None, "patchifier must be provided to get n_position_axes"

        device = torch.device("cuda")

        drop_probs = torch.rand(len(data_batch.raw_texts)).tolist()
        for i, drop_prob in enumerate(drop_probs):
            if drop_prob < txt_drop_prob:
                data_batch.raw_texts[i] = ""

        # Auto-infer n_position_axes from patchifier
        n_position_axes = self.lfm.patchifier.get_num_position_axes()

        """Encode text into embeddings (in-place)."""
        if self.lfm.text_encoder is not None:
            with fwd_only_mode(self.lfm.text_encoder):
                data_batch.txt, data_batch.txt_datum_lens = self.lfm.text_encoder(data_batch.raw_texts)
                data_batch.txt_position_ids = torch.zeros(
                    data_batch.txt.shape[0], n_position_axes, device=device, dtype=torch.int32
                )  # type: ignore

        if self.lfm.clip_encoder is not None:
            # clip is (b, d)
            with fwd_only_mode(self.lfm.clip_encoder):
                data_batch.vec = self.lfm.clip_encoder(data_batch.raw_texts)

        """Encode image into embeddings (in-place)."""
        if self.lfm.patchifier is not None and self.lfm.vae is not None:
            if data_batch.raw_latents is None:
                assert data_batch.raw_images is not None, "raw_images must be provided"
                data_batch.raw_images = data_batch.raw_images.to(device=device)
                with fwd_only_mode(self.lfm.vae):
                    data_batch.raw_latents = self.lfm.vae.encode(data_batch.raw_images)
            else:
                data_batch.raw_latents = data_batch.raw_latents.to(device=device)

            data_batch.img_clean, data_batch.img_datum_lens, data_batch.img_position_ids = self.lfm.patchifier.patchify(
                data_batch.raw_latents
            )
            assert (
                data_batch.img_position_ids.shape[1] == n_position_axes
            ), f"img_position_ids must have {n_position_axes} axes, got {data_batch.img_position_ids.shape[1]}"

        """Add noise to the image (in-place)."""
        if (
            data_batch.img_clean is not None
            and data_batch.img_datum_lens is not None
            and self.lfm.time_sampler is not None
            and self.lfm.time_warper is not None
            and self.lfm.time_weighter is not None
            and self.lfm.flow_noiser is not None
        ):
            timesteps = self.lfm.time_sampler((len(data_batch.raw_texts),), device=device)

            timesteps = self.lfm.time_warper(timesteps, data_batch.img_datum_lens)
            data_batch.timesteps = timesteps

            data_batch.timestep_weights = self.lfm.time_weighter(data_batch.timesteps)

            data_batch.img, data_batch.img_v_truth = self.lfm.flow_noiser(
                data_batch.img_clean, data_batch.img_datum_lens, data_batch.timesteps
            )

        if data_batch.txt is not None and data_batch.img is not None:
            data_batch.num_tokens = data_batch.txt.shape[0] + data_batch.img.shape[0]

        return data_batch


@dataclass
class TrainableOps:
    lfm: LatentFM
    global_batch_size: int | None = None

    def __call__(self, data_batch: FMDataContext) -> FMDataContext:
        """Pass through the denoiser with correct interface"""
        assert data_batch.txt is not None, "txt must be provided"
        assert data_batch.txt_datum_lens is not None, "txt_datum_lens must be provided"
        assert data_batch.txt_position_ids is not None, "txt_position_ids must be provided"
        assert data_batch.img is not None, "img must be provided"
        assert data_batch.img_datum_lens is not None, "img_datum_lens must be provided"
        assert data_batch.img_position_ids is not None, "img_position_ids must be provided"
        assert data_batch.timesteps is not None, "timesteps must be provided"
        assert self.lfm.denoiser is not None, "denoiser must be provided"

        data_batch.img_v_pred = self.lfm.denoiser(
            txt=data_batch.txt,
            txt_datum_lens=data_batch.txt_datum_lens,
            txt_position_ids=data_batch.txt_position_ids,
            img=data_batch.img,
            img_datum_lens=data_batch.img_datum_lens,
            img_position_ids=data_batch.img_position_ids,
            t=data_batch.timesteps,
            vec=data_batch.vec,
            sequence_balancer=self.lfm.dit_balancer,
        )

        assert data_batch.img_v_pred is not None, "img_v_pred must be provided"
        assert data_batch.img_v_truth is not None, "img_v_truth must be provided"
        assert data_batch.img_datum_lens is not None, "img_datum_lens must be provided"
        assert data_batch.timestep_weights is not None, "timestep_weights must be provided"

        mse_loss = (data_batch.img_v_pred.float() - data_batch.img_v_truth.float()) ** 2
        loss_vec = pack_reduce(mse_loss, data_batch.img_datum_lens, reduction="mean")
        loss_vec = loss_vec * data_batch.timestep_weights
        if self.global_batch_size is None:
            device = torch.device("cuda")
            bs = torch.tensor(len(data_batch.img_datum_lens), device=device)
            dist.all_reduce(bs, op=dist.ReduceOp.SUM)
            self.global_batch_size = int(bs.item())
        data_batch.loss = loss_vec.sum() * dist.get_world_size() / self.global_batch_size
        data_batch.loss_vec = loss_vec
        return data_batch


class VelocityModel:
    """Wrapper for velocity prediction models with support for classifier-free guidance.

    This class encapsulates a velocity prediction model (typically a transformer) along with
    text conditioning. For classifier-free guidance, users should provide concatenated
    positive and negative conditioning in the initialization parameters.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        txt: torch.Tensor,
        txt_datum_lens: torch.Tensor,
        txt_position_ids: torch.Tensor,
        vec: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        sequence_balancer: SequenceBalancer | None = None,
        guidance: float | None = None,
    ) -> None:
        """Initialize the velocity model wrapper.

        Args:
            denoiser: The underlying neural network model that predicts velocity.
            txt: Text conditioning tokens. Contains tokens for n sequences (non-CFG) or
                 2*n sequences (CFG) where n is the image batch size. For CFG, contains
                 positive conditioning followed by negative conditioning concatenated together.
            txt_datum_lens: Length of each text sequence in the batch. Should contain n elements
                           for non-CFG or 2*n elements for CFG (where n is the image batch size).
                           For CFG, contains lengths for positive conditioning followed by negative conditioning.
            txt_position_ids: Position IDs for text sequences. Contains position IDs for
                             n sequences (non-CFG) or 2*n sequences (CFG) where n is the image batch size.
                             For CFG, contains positive conditioning followed by negative conditioning.
            vec: Optional additional vector conditioning (e.g., style vectors).
                 Shape: (n, d_vec) for non-CFG or (2*n, d_vec) for CFG, where n is the
                 image batch size and d_vec is vector dimension. For CFG, this should
                 contain positive conditioning followed by negative conditioning.
            cfg_scale: Classifier-free guidance scale. Values > 1.0 enable CFG,
                      with higher values increasing conditioning strength.
            guidance: Optional, specially for flux model. Use for distilled guidance embedding.
        """
        super().__init__()
        self.denoiser = denoiser
        self.txt = txt
        self.txt_datum_lens = txt_datum_lens
        self.txt_position_ids = txt_position_ids
        self.vec = vec
        self.cfg_scale = cfg_scale
        self.sequence_balancer = sequence_balancer
        self.guidance = None
        if guidance is not None:
            self.guidance = torch.tensor(guidance, dtype=vec.dtype, device=vec.device).repeat(len(txt_datum_lens))

    def __call__(
        self,
        img: torch.Tensor,
        img_datum_lens: torch.Tensor,
        img_position_ids: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field for the given images with optional classifier-free guidance.

        Args:
            img: Image tokens/latents in packed sequence format.
                 Shape: (l1+l2+...+ln, d_model) where l1, l2, ..., ln are the lengths
                 of individual sequences and d_model is the model dimension.
            img_datum_lens: Length of each image sequence in the batch, shape (n,).
            img_position_ids: Position IDs in packed sequence format.
                             Shape: (l1+l2+...+ln, d_position) where d_position is
                             the number of position axes.
            t: Time steps for the flow matching process, shape (n,).

        Note:
            Both img and txt use packed sequence format where variable-length sequences
            are concatenated together with corresponding length tensors.
            For CFG (cfg_scale > 1.0), the following batch size relationships are expected:
            - txt_datum_lens should have 2*n elements (where n = len(img_datum_lens))
            - vec should have shape (2*n, d_vec) if provided
            Both contain positive conditioning followed by negative conditioning.

        Returns:
            Predicted velocity field for the images in packed format
        """

        txt, txt_datum_lens, txt_position_ids, vec = self.txt, self.txt_datum_lens, self.txt_position_ids, self.vec

        if self.cfg_scale > 1.0:
            if len(txt_datum_lens) != 2 * len(img_datum_lens):
                raise ValueError(
                    f"For classifier-free guidance (cfg_scale > 1.0), txt_datum_lens must have "
                    f"2*n elements where n = len(img_datum_lens). Got txt_datum_lens length: "
                    f"{len(txt_datum_lens)}, expected: {2 * len(img_datum_lens)} (2 * {len(img_datum_lens)})"
                )

            # Duplicate img data for positive and negative conditioning
            img = torch.cat([img, img], dim=0)
            img_datum_lens = torch.cat([img_datum_lens, img_datum_lens], dim=0)
            img_position_ids = torch.cat([img_position_ids, img_position_ids], dim=0)
            t = torch.cat([t, t], dim=0)

        img_v: torch.Tensor = self.denoiser(
            txt=txt,
            txt_datum_lens=txt_datum_lens,
            txt_position_ids=txt_position_ids,
            img=img,
            img_datum_lens=img_datum_lens,
            img_position_ids=img_position_ids,
            t=t,
            vec=vec,
            guidance=self.guidance,
            sequence_balancer=self.sequence_balancer,
        )

        if self.cfg_scale > 1.0:
            # Split results and apply energy-preserving CFG
            img_datum_lens = img_datum_lens.chunk(2, dim=0)[0]
            pos_img_v, neg_img_v = img_v.chunk(2, dim=0)
            img_v = energy_preserve_cfg(pos_img_v, neg_img_v, img_datum_lens, self.cfg_scale)

        return img_v


@dataclass
class InferenceTask:
    img_fhw: tuple[int, int, int]  # frame, height, width
    prompts: list[str]
    neg_prompts: list[str]
    cfg_scale: float
    num_steps: int
    eta: float
    seed: int
    output_names: list[str]
    guidance: float | None = None


def load_prompts_as_tasks(
    img_fhw: tuple[int, int, int],
    prompt_file: str,
    samples_per_prompt: int = 2,
    neg_prompt: str = "",
    cfg_scale: float = 5,
    num_steps: int = 50,
    eta: float = 1.0,
    file_ext: str = "jpg",
    per_gpu_bs: int = 16,
    guidance: float | None = None,
) -> list[InferenceTask]:
    prompts = []
    with open(prompt_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    # add dummy prompt so that it's divisible by dist.get_world_size()
    num_gpus = dist.get_world_size()
    while len(prompts) % (num_gpus * per_gpu_bs) != 0:
        prompts.append("DUMMY_PROMPT")

    tasks = []
    for seed in range(samples_per_prompt):
        for i in range(0, len(prompts), per_gpu_bs):
            tasks.append(
                InferenceTask(
                    img_fhw=img_fhw,
                    prompts=prompts[i : i + per_gpu_bs],
                    neg_prompts=[neg_prompt] * per_gpu_bs,
                    cfg_scale=cfg_scale,
                    num_steps=num_steps,
                    eta=eta,
                    seed=seed,
                    output_names=[f"p{k:06d}-s{seed:06d}.{file_ext}" for k in range(i, i + per_gpu_bs)],
                    guidance=guidance,
                )
            )

    # divide to each gpu
    num_tasks_per_gpu = len(tasks) // num_gpus
    gpu_rank = dist.get_rank()
    tasks = tasks[gpu_rank * num_tasks_per_gpu : (gpu_rank + 1) * num_tasks_per_gpu]
    return tasks


@dataclass
class InferenceOps:
    lfm: LatentFM

    @torch.no_grad()
    def __call__(
        self,
        output_dir: str,
        img_fhw: tuple[int, int, int],
        prompt_file: str,
        samples_per_prompt: int = 2,
        neg_prompt: str = "",
        cfg_scale: float = 5,
        num_steps: int = 50,
        eta: float = 1.0,
        file_ext: str = "jpg",
        per_gpu_bs: int = 16,
        use_ema: bool = True,
        guidance: float | None = None,
        sample_method: Literal["euler", "ddim"] = "ddim",
        save_as_npz: bool = False,
    ) -> None:
        """
        Perform flow matching inference sampling.

        Args:
            save_as_npz: If True, saves all images as a uint8 npz file instead of individual image files.
                        Also saves metadata and sampling info as a JSON file.
        """
        # Select denoiser based on use_ema flag
        selected_denoiser = self.lfm.denoiser
        if use_ema and self.lfm.ema_denoiser is not None:
            # if self.lfm.ema_denoiser is None:
            #     raise ValueError("EMA denoiser is not available, please use non-EMA denoiser")
            selected_denoiser = self.lfm.ema_denoiser
            logger.info("Using EMA denoiser for inference")

        assert selected_denoiser is not None, "denoiser must be provided"
        assert self.lfm.vae is not None, "vae must be provided"
        assert self.lfm.patchifier is not None, "patchifier must be provided"
        assert self.lfm.text_encoder is not None, "text_encoder must be provided"
        assert self.lfm.flow_noiser is not None, "flow_noiser must be provided"
        assert self.lfm.time_warper is not None, "time_warper must be provided"

        with fwd_only_mode(selected_denoiser):
            device = next(selected_denoiser.parameters()).device

            tasks = load_prompts_as_tasks(
                img_fhw,
                prompt_file,
                samples_per_prompt,
                neg_prompt,
                cfg_scale,
                num_steps,
                eta,
                file_ext,
                per_gpu_bs,
                guidance=guidance,
            )

            os.makedirs(os.path.join(output_dir, "assets"), exist_ok=True)
            asset_metadata_list: list[dict[str, Any]] = []
            all_images_for_npz: list[torch.Tensor] = []
            for task in tqdm(tasks, desc="Generating images", disable=dist.get_rank() != 0):
                # Set random seed for reproducible generation
                generator = torch.Generator(device=device).manual_seed(task.seed)

                # Encode text prompt
                with fwd_only_mode(self.lfm.text_encoder):
                    txt, txt_datum_lens = self.lfm.text_encoder(task.prompts)

                # Auto-infer n_position_axes from patchifier
                n_position_axes = self.lfm.patchifier.get_num_position_axes()
                txt_position_ids = torch.zeros(txt.shape[0], n_position_axes, device=device, dtype=torch.int32)

                # Calculate latent dimensions after VAE encoding using Patchifier properties
                vae_c = self.lfm.patchifier.vae_latent_channels
                vae_cf, vae_ch, vae_cw = self.lfm.patchifier.vae_compression_factors

                img_f, img_h, img_w = task.img_fhw
                latent_f = int(img_f / vae_cf)
                latent_h = int(img_h / vae_ch)
                latent_w = int(img_w / vae_cw)

                # Create noise tensor in latent space shape (b, vae_c, f/vae_cf, h/vae_ch, w/vae_cw)
                noise_tensor = torch.randn(
                    len(task.prompts),
                    vae_c,
                    latent_f,
                    latent_h,
                    latent_w,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )

                # Patchify the noise tensor to get patches
                x_noise, img_datum_lens, img_position_ids = self.lfm.patchifier.patchify(noise_tensor)

                # Add negative prompts if cfg_scale > 1.0
                if cfg_scale > 1.0:
                    neg_txt, neg_txt_datum_lens = self.lfm.text_encoder(task.neg_prompts)
                    txt = torch.cat([txt, neg_txt], dim=0)
                    txt_datum_lens = torch.cat([txt_datum_lens, neg_txt_datum_lens], dim=0)
                    neg_txt_position_ids = torch.zeros(
                        neg_txt.shape[0], n_position_axes, device=device, dtype=torch.int32
                    )
                    txt_position_ids = torch.cat([txt_position_ids, neg_txt_position_ids], dim=0)

                # Add CLIP conditioning if available
                vec = None
                if self.lfm.clip_encoder is not None:
                    with fwd_only_mode(self.lfm.clip_encoder):
                        vec = self.lfm.clip_encoder(task.prompts)
                        if cfg_scale > 1.0:
                            neg_vec = self.lfm.clip_encoder(task.neg_prompts)
                            vec = torch.cat([vec, neg_vec], dim=0)

                # Create velocity model wrapper
                velocity_model = VelocityModel(
                    denoiser=selected_denoiser,
                    txt=txt,
                    txt_datum_lens=txt_datum_lens,
                    txt_position_ids=txt_position_ids,
                    vec=vec,
                    cfg_scale=cfg_scale,
                    sequence_balancer=self.lfm.dit_balancer,
                    guidance=task.guidance,
                )

                # Create flow sampler
                flow_sampler = FlowSampler(
                    velocity_model=velocity_model,
                    noiser=cast(NoiserProtocol, self.lfm.flow_noiser),
                    t_warper=self.lfm.time_warper,
                    sample_method=sample_method,
                )

                # Perform sampling with appropriate parameters
                warp_len = int(img_datum_lens[0].item())  # Use patch count for time warping

                # Run the denoising loop
                x_clean = flow_sampler(
                    x=x_noise,
                    x_datum_lens=img_datum_lens,
                    x_position_ids=img_position_ids,
                    num_steps=task.num_steps,
                    warp_len=warp_len,
                    rng=generator,
                    eta=task.eta,
                )

                # Unpatchify to get latent tensors
                latents = self.lfm.patchifier.unpatchify(x_clean)  # (b, c, f, h, w)

                # Decode with VAE to get final images
                with fwd_only_mode(self.lfm.vae):
                    images = self.lfm.vae.decode(latents)  # type: ignore

                # Convert to uint8 [0, 255] range
                images = (images + 1) * 127.5  # Convert to [0, 255]
                images = images.round().clamp(0, 255).to(torch.uint8)

                for prompt, img, output_name in zip(task.prompts, images, task.output_names, strict=True):
                    if prompt == "DUMMY_PROMPT":
                        continue

                    # (c, f, h, w) -> (f, h, w, c) -> (h, w, c)
                    img_processed = img.permute(1, 2, 3, 0)[0]

                    if not save_as_npz:
                        # Save individual image files (default behavior)
                        img_pil = Image.fromarray(img_processed.cpu().numpy())
                        img_pil.save(os.path.join(output_dir, "assets", output_name))
                    else:
                        # Additionally collect for NPZ if requested
                        all_images_for_npz.append(img_processed.cpu())

                    asset_metadata_list.append(
                        {
                            "path": os.path.join("assets", output_name),
                            "prompt": prompt,
                        }
                    )

        # Make a index html file
        gathered_asset_metadata_list = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_asset_metadata_list, asset_metadata_list)
        gathered_asset_metadata_list_flat: list[dict[str, Any]] = list(
            itertools.chain.from_iterable(gathered_asset_metadata_list)
        )
        gathered_asset_metadata_list_flat.sort(key=lambda x: x["path"])  # sort by path

        # Prepare sampling info
        sampling_info = {
            "use_ema": use_ema,
            "prompt_file": prompt_file,
            "img_fhw": img_fhw,
            "samples_per_prompt": samples_per_prompt,
            "neg_prompt": neg_prompt,
            "cfg_scale": cfg_scale,
            "num_steps": num_steps,
            "eta": eta,
            "per_gpu_bs": per_gpu_bs,
            "file_ext": file_ext,
            "save_as_npz": save_as_npz,
        }
        metadata = {
            "sampling_info": sampling_info,
            "asset_metadata": gathered_asset_metadata_list_flat,
            "total_images": len(gathered_asset_metadata_list_flat),
        }

        if save_as_npz:
            gathered_images = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_images, all_images_for_npz)

        # Generate HTML gallery (default behavior)
        if dist.get_rank() == 0:
            # Save metadata as JSON
            with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved metadata to metadata.json")

            # Generate HTML gallery
            if not save_as_npz:
                generate_html_gallery(
                    output_dir=output_dir,
                    asset_metadata_list=gathered_asset_metadata_list_flat,
                    sampling_info=sampling_info,
                    images_per_row=samples_per_prompt * 2,
                )
            else:
                all_images = list(itertools.chain.from_iterable(gathered_images))
                images_array = torch.stack(all_images, dim=0).numpy().astype(np.uint8)  # (n, h, w, c)
                np.savez_compressed(os.path.join(output_dir, "assets/images.npz"), images_array, allow_pickle=False)
                logger.info(f"Saved {images_array.shape[0]} images to {os.path.join(output_dir, 'assets/images.npz')}")

        dist.barrier()
