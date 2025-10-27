from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
import gc
import os
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

# Local imports
from data import DataStreamer
from models.latent_fm import FMDataContext, FrozenOps, InferenceOps, LatentFM, TrainableOps
from models.latent_fm_factory import create_latent_fm
from utils.ckpt import FSDPCheckpointer, load_checkpoint_for_inference
from utils.clip_grad import clip_grad
from utils.config import ConfigurableModule
from utils.ema import copy_params, update_ema
from utils.fsdp import fwd_only_mode
from utils.log import TrackingLogger, WandbLogger, get_logger, get_pbar, human_readable_number
from utils.lr import LinearWarmupCosineDecayScheduler
from utils.optim import create_optimizer
from utils.prof import Profiler

from . import BaseTrainerParams, setup_distributed, setup_experiment_dirs

logger = get_logger(__name__)


@dataclass
class DITTrainerParams(BaseTrainerParams):
    """Parameters for DiTTrainer."""

    # Text dropout probability
    txt_drop_prob: float = 0.1


class DiTTrainer(ConfigurableModule[DITTrainerParams]):
    """Distributed Flow Matching trainer."""

    # ---- Components used for training ----
    params: DITTrainerParams | None = None
    config: dict[str, Any] | None = None

    latent_fm: LatentFM | None = None
    frozen_ops: FrozenOps | None = None
    trainable_ops: TrainableOps | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    checkpointer: FSDPCheckpointer | None = None

    device: torch.device | None = None
    local_rank: int | None = None
    global_rank: int | None = None
    world_size: int | None = None

    exp_dir: str | None = None
    run_dir: str | None = None
    ckpt_dir: str | None = None

    step: int | None = None
    start_step: int | None = None

    def __init__(self, params: DITTrainerParams):
        self.params = params
        logger.info(f"Initialized DiTTrainer with params: {params}")

    @classmethod
    def get_default_params(cls) -> DITTrainerParams:
        """Return the default parameters for DiTTrainer."""
        return DITTrainerParams()

    @property
    def training_state(self) -> dict[str, Any]:
        return {
            "model": self.latent_fm.denoiser,
            "ema": self.latent_fm.ema_denoiser,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

    @property
    def training_meta(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "world_size": self.world_size,
            "run_dir": self.run_dir,
            "ckpt_dir": self.ckpt_dir,
        }

    @property
    def denoiser(self) -> nn.Module:
        return self.latent_fm.denoiser

    @property
    def ema_denoiser(self) -> nn.Module:
        return self.latent_fm.ema_denoiser

    def train(self, config: dict[str, Any]) -> None:
        """Train the model."""
        self.config = config

        # Initialize training
        self.train_init()

        # Run training loop
        self.train_loop()

        # Clean up distributed training
        dist.destroy_process_group()

    def train_init(self) -> None:
        # Use rank-invariant seed up util the creation of dataloader
        torch.manual_seed(0)

        # Initialize distributed training
        self.device, self.local_rank, self.global_rank, self.world_size = setup_distributed()

        # Setup experiment directories
        self.exp_dir = self.params.exp_dir
        self.run_dir, self.ckpt_dir = setup_experiment_dirs(self.exp_dir, self.config)

        # Create LatentFM with all components
        logger.info("Creating LatentFM with all components...")
        self.latent_fm = create_latent_fm(self.config, self.device)

        # Ensure the required trainable modules are present (type safety for mypy)
        assert self.latent_fm.denoiser is not None, "Denoiser must be provided by create_latent_fm"
        assert self.latent_fm.ema_denoiser is not None, "EMA denoiser must be provided by create_latent_fm"

        # Create FrozenOps and TrainableOps
        logger.info("Setting up FrozenOps and TrainableOps...")
        self.frozen_ops = FrozenOps(lfm=self.latent_fm)
        self.trainable_ops = TrainableOps(lfm=self.latent_fm, global_batch_size=None)

        # Create optimizer (only for trainable denoiser parameters)
        logger.info("Setting up optimizer...")
        self.optimizer = create_optimizer(
            model=self.denoiser,
            use_muon=self.params.use_muon,
            adam_lr=self.params.adam_max_lr,
            adam_betas=self.params.adam_betas,
            muon_param_patterns=self.params.muon_param_patterns,
            muon_lr=self.params.muon_max_lr,
            muon_mu=self.params.muon_mu,
            muon_adjust_lr=self.params.muon_adjust_lr,
            weight_decay=self.params.weight_decay,
        )

        # Create learning rate scheduler
        logger.info("Setting up learning rate scheduler...")
        self.scheduler = LinearWarmupCosineDecayScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.params.warmup_steps,
            total_steps=self.params.max_steps,
            min_lr_ratio=self.params.min_lr_ratio,
        )

        # Switch to rank-dependent seed
        # Note that checkpointer may restore RNG state from checkpoint if it exists
        torch.manual_seed(42 + self.global_rank)

        # Initialize checkpointer
        logger.info("Setting up checkpointer...")
        self.checkpointer = FSDPCheckpointer(self.ckpt_dir)

        # Resume from latest checkpoint if available
        logger.info("Trying to resume from latest checkpoint...")
        self.start_step = self.checkpointer.resume_latest(
            **self.training_state,
            init_ckpt=self.params.init_ckpt,
            init_ckpt_load_plan=self.params.init_ckpt_load_plan,
        )
        if self.start_step == 0:
            # Initialize EMA with main model parameters
            logger.info("Initializing EMA with main model parameters...")
            copy_params(self.denoiser, self.ema_denoiser, model_to_ema=True)
        self.step = self.start_step

        # Inference loop
        if self.params.inference_at_start:
            logger.info(f"Running inference at start step {self.start_step}...")
            self.in_train_inference()
            if self.params.inference_then_exit:
                logger.info("Finished inference and not continuing training, exiting...")
                dist.destroy_process_group()
                return

        # Initialize wandb logger
        self.wandb_logger = WandbLogger(
            project=self.params.wandb_project,
            name=self.params.wandb_name,
            config=self.training_meta,
            entity=self.params.wandb_entity,
            host=self.params.wandb_host,
            save_dir=self.run_dir,
            mode=self.params.wandb_mode,
        )

        # Create infinite dataloader
        logger.info("Setting up data streamer...")
        self.data_module = DataStreamer(self.config["data"])
        self.train_dataloader = self.data_module.train_dataloader()

    def train_loop(self) -> None:
        """Run the training loop."""
        # Main training loop
        logger.info("Running memory cleanup before entering training loop...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        gc.disable()

        self.denoiser.train()
        self.ema_denoiser.eval()  # EMA model should be in eval mode
        self.optimizer.zero_grad()
        self.tracking_logger = TrackingLogger()

        # Initialize progress bar for rank 0
        pbar = get_pbar(self.params.max_steps, self.start_step)

        logger.info(f"Starting training from step {self.start_step} for {self.params.max_steps} total steps")
        while self.step < self.params.max_steps:
            self.train_one_step()

            # Update progress bar for rank 0
            if pbar is not None:
                pbar.update(1)

        # Save final checkpoint
        if self.step % self.params.ckpt_freq != 0:  # Avoid duplicate checkpoint saving
            self.save_checkpoint()
        self.checkpointer.finish()

        # Close progress bar
        if pbar is not None:
            pbar.close()

    def train_one_step(self) -> None:
        self.step += 1

        prof = None
        dump_trace = self.step == self.start_step + 10
        if dump_trace:
            prof = Profiler()
            prof.start()

        train_step_tic = time.time()

        self.run_fwdbwd(
            dataloader=self.train_dataloader,
            tracking_logger=self.tracking_logger,
            total_batch_size=self.params.total_batch_size,
            print_data_summary=(self.step == self.start_step),
            txt_drop_prob=self.params.txt_drop_prob,
        )

        # Gradient clipping
        grad_norm = clip_grad(self.denoiser, self.params.gradient_clip_norm)

        # Gradient norm spike detection
        has_bad_grad = (not torch.isfinite(grad_norm)) or (
            self.step >= self.params.grad_norm_spike_detection_start_step
            and grad_norm > self.params.grad_norm_spike_threshold
        )
        if has_bad_grad and self.global_rank == 0:
            logger.warning(
                f"Step {self.step}: detected nan/inf/spiky gradient (norm={grad_norm}); skipping optimizer step."
            )

        # Optimizer step (only if gradient is healthy)
        lrs = self.scheduler.get_last_lr()  # List of learning rates for each param group
        if not has_bad_grad:
            self.optimizer.step()
            update_ema(self.denoiser, self.ema_denoiser, self.params.ema_decay)

        # Zero out gradients
        self.optimizer.zero_grad()

        # Always step the scheduler regardless of whether optimizer step was skipped
        self.scheduler.step()

        # Update tracking logger
        log_dict = {
            "has_bad_grad": has_bad_grad,
            "grad_norm": grad_norm,
            "step_duration": time.time() - train_step_tic,
        }

        # Log learning rates for each parameter group
        if len(lrs) == 1:
            log_dict["lr"] = lrs[0]
        else:
            # Multiple parameter groups - log each separately
            for i, lr in enumerate(lrs):
                log_dict[f"lr_{i}"] = lr

        self.tracking_logger.log(log_dict)

        if dump_trace:
            trace_path = (
                os.path.join(self.run_dir, f"traces/rank_{self.global_rank}.json") if self.global_rank == 0 else None
            )
            prof.stop(trace_path)

        # Print global batch size
        if self.step == self.start_step + 1:
            logger.info(f"Global batch size: {self.trainable_ops.global_batch_size}")

        # Periodic garbage collection and synchronization across GPUs to prevent stragglers
        if self.step % self.params.gc_freq == 0:
            logger.info(f"Step {self.step:6d} | Running garbage collection...")
            gc.collect()

        # Logging
        if self.step % self.params.log_freq == 0:
            self.log_metrics()

        # Save checkpoint
        if self.step % self.params.ckpt_freq == 0:
            self.save_checkpoint()

        # Inference loop
        if self.step % self.params.inference_freq == 0:
            self.in_train_inference()

        # Validation loop
        if self.step % self.params.val_freq == 0:
            self.run_validation()

    def run_fwdbwd(
        self,
        dataloader: Iterator[FMDataContext],
        tracking_logger: TrackingLogger,
        total_batch_size: int = -1,
        skip_backward: bool = False,
        print_data_summary: bool = False,
        txt_drop_prob: float = 0.1,
    ) -> torch.Tensor:
        n_accum_steps = 0
        while True:
            # Get next batch (FMDataContext) - infinite iterator, no StopIteration
            with tracking_logger.log_time("time/data"):
                fm_data_context = next(dataloader)

            # Step 1: Process through FrozenOps (T5, VAE, Patchifier, FlowNoiser)
            with tracking_logger.log_time("time/frozen_ops"):
                fm_data_context = self.frozen_ops(fm_data_context, txt_drop_prob=txt_drop_prob)

                # Print data summary if requested
                if print_data_summary:
                    fm_data_context.summarize()

            # Step 2: Process through TrainableOps (Denoiser + Loss computation)
            with tracking_logger.log_time("time/trainable_ops_fwd"):
                fm_data_context = self.trainable_ops(fm_data_context)

            global_batch_size = self.trainable_ops.global_batch_size
            if total_batch_size <= 0:
                target_n_accum_steps = 1
            else:
                target_n_accum_steps = (total_batch_size + global_batch_size - 1) // global_batch_size

            if not skip_backward:
                with tracking_logger.log_time("time/trainable_ops_bwd"):
                    (fm_data_context.loss / target_n_accum_steps).backward()

            tracking_logger.log({"loss_vec": fm_data_context.loss_vec, "num_tokens": fm_data_context.num_tokens})

            n_accum_steps += 1
            if n_accum_steps >= target_n_accum_steps:
                break

    @torch.no_grad()
    def run_validation(self) -> None:
        self.denoiser.eval()

        logger.info(f"Step {self.step:6d} | Running validation...")

        val_dataloader = self.data_module.val_dataloader()
        val_tracking_logger = TrackingLogger()

        with fwd_only_mode(self.denoiser), val_tracking_logger.log_time("time/validation"):
            self.run_fwdbwd(
                dataloader=val_dataloader,
                tracking_logger=val_tracking_logger,
                total_batch_size=self.params.val_num_samples,
                skip_backward=True,
                txt_drop_prob=0.0,
            )

        val_tracking_logger.flush()
        avg_val_loss = val_tracking_logger["loss_vec", "mean"]
        max_val_time = val_tracking_logger["time/validation", "max"]

        logger.info(f"Step {self.step:6d} | Validation loss: {avg_val_loss:.4f}")

        if self.global_rank == 0:
            self.wandb_logger.log({"validation/loss": avg_val_loss, "time/validation": max_val_time}, step=self.step)

    def save_checkpoint(self) -> None:
        self.checkpointer.save_checkpoint(
            step=self.step,
            **self.training_state,
            metadata=self.training_meta,
        )
        self.checkpointer.finish()

    def log_metrics(self) -> None:
        # Compute average loss across all batch elements and GPUs
        self.tracking_logger.flush()
        avg_loss = self.tracking_logger["loss_vec", "mean"]

        # Handle learning rates (might be single or multiple)
        lr_metrics = {}
        if "lr" in self.tracking_logger.stats:
            lr_metrics["lr"] = self.tracking_logger["lr", "mean"]
        else:
            # Multiple learning rates
            for key in self.tracking_logger.stats:
                if key.startswith("lr_"):
                    lr_metrics[key] = self.tracking_logger[key, "mean"]

        max_grad_norm = self.tracking_logger["grad_norm", "max"]
        bad_grad_count = self.tracking_logger["has_bad_grad", "sum"]
        tps = self.tracking_logger["num_tokens", "sum"] / self.tracking_logger["step_duration", "sum"] * self.world_size
        max_data_time = self.tracking_logger["time/data", "max"]
        max_frozen_ops_time = self.tracking_logger["time/frozen_ops", "max"]
        max_trainable_ops_fwd_time = self.tracking_logger["time/trainable_ops_fwd", "max"]
        max_trainable_ops_bwd_time = self.tracking_logger["time/trainable_ops_bwd", "max"]

        if dist.get_rank() == 0:
            # Format learning rate display
            if "lr" in lr_metrics:
                lr_display = f"LR: {lr_metrics['lr']:.2e}"
            else:
                lr_values = [f"{lr:.2e}" for key, lr in sorted(lr_metrics.items())]
                lr_display = f"LRs: [{', '.join(lr_values)}]"

            logger.info(
                f"Step {self.step:6d} | Loss: {avg_loss:.4f} | {lr_display} | GradNorm: {max_grad_norm:.2f} | "
                f"TPS: {human_readable_number(tps)} | Data: {max_data_time:.3f}s | Frozen: {max_frozen_ops_time:.3f}s | "
                f"TrainableFwd: {max_trainable_ops_fwd_time:.3f}s | TrainableBwd: {max_trainable_ops_bwd_time:.3f}s"
            )

            # Prepare wandb metrics
            wandb_metrics = {
                "train/loss": avg_loss,
                "train/gradient_norm": max_grad_norm,
                "train/bad_grad_count": bad_grad_count,
                "train/tps": tps,
                "time/data": max_data_time,
                "time/frozen_ops": max_frozen_ops_time,
                "time/trainable_ops_fwd": max_trainable_ops_fwd_time,
                "time/trainable_ops_bwd": max_trainable_ops_bwd_time,
            }

            # Add learning rate(s) to wandb metrics
            if "lr" in lr_metrics:
                wandb_metrics["train/learning_rate"] = lr_metrics["lr"]
            else:
                for key, lr in lr_metrics.items():
                    wandb_metrics[f"train/learning_rate_{key.split('_')[1]}"] = lr

            self.wandb_logger.log(wandb_metrics, step=self.step)

    def in_train_inference(self) -> None:
        assert self.latent_fm is not None and self.exp_dir is not None and self.step is not None, (
            "LatentFM, exp_dir, and step must be provided in training mode"
        )

        logger.info("Setting up InferenceOps...")
        inference_ops = InferenceOps(lfm=self.latent_fm)

        for use_ema in [False, True]:
            logger.info(f"Running inference with EMA={int(use_ema)}...")
            config = deepcopy(self.config)
            config["inferencer"]["inference_ops_args"]["use_ema"] = use_ema
            config["inferencer"]["inference_ops_args"]["output_dir"] = os.path.join(
                self.exp_dir, f"inference_ema-{int(use_ema)}/step_{self.step:08d}"
            )
            inference_ops(**config["inferencer"]["inference_ops_args"])

    def inference(self, config: dict[str, Any]) -> None:
        self.config = config

        # Use rank-invariant seed up util the creation of dataloader
        torch.manual_seed(0)

        # Initialize distributed training
        self.device, self.local_rank, self.global_rank, self.world_size = setup_distributed()

        # Create LatentFM with all components
        logger.info("Creating LatentFM with all components...")
        self.latent_fm = create_latent_fm(self.config, self.device, create_ema=False)

        # Ensure the required modules are present (type safety for mypy)
        assert self.latent_fm.denoiser is not None, "Denoiser must be provided by create_latent_fm"

        ckpt_dir = config["inferencer"]["ckpt_dir"]
        use_ema = config["inferencer"]["inference_ops_args"]["use_ema"]
        model_key = "ema" if use_ema else "model"
        logger.info(f"Loading checkpoint from {ckpt_dir} with model_key={model_key}...")
        load_checkpoint_for_inference(ckpt_dir, self.denoiser, model_key=model_key)

        logger.info("Setting up InferenceOps...")
        inference_ops = InferenceOps(lfm=self.latent_fm)

        logger.info("Running inference...")
        inference_ops(**config["inferencer"]["inference_ops_args"])

        dist.destroy_process_group()