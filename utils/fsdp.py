from contextlib import contextmanager
import functools
from operator import attrgetter
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
import torch.nn as nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
    noop_context_fn,
)

from utils.log import get_logger
from utils.misc import DTYPE_MAP

logger = get_logger(__name__)


def _apply_ac_to_block(module: nn.Module, ac_freq: int, save_attn_output: bool = False) -> nn.Module:
    # Checkpoint every `ac_freq` of the modules passed to this function
    context_fn = noop_context_fn

    if save_attn_output:

        def policy_fn(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
            # Force exclude flash attention from checkpointing
            ops_to_save = [torch.ops.flash_attn._flash_attn_varlen_forward.default]
            if op in ops_to_save:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_RECOMPUTE

        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)

    ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
    ptd_checkpoint_wrapper._count += 1  # type: ignore
    if (ac_freq > 0) and (ptd_checkpoint_wrapper._count % ac_freq == 0):  # type: ignore
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False, context_fn=context_fn)
    else:
        return module


def apply_ac(model: nn.Module, ac_freq: int, blocks_attr: str | list[str] = "blocks") -> None:
    """
    Modified from: https://github.com/pytorch/torchtitan/blob/7d5f3cc698853d2227cf5433776406d0e0345424/torchtitan/models/llama3/infra/parallelize.py#L303

    Apply activation checkpointing to the model.

    Args:
        model: The model to apply activation checkpointing to
        ac_freq: Checkpointing frequency (every ac_freq blocks)
        blocks_attr: Name of the attribute containing the model blocks (e.g., "blocks", "layers")
    """
    if isinstance(blocks_attr, str):
        blocks_attr = [blocks_attr]

    for attr in blocks_attr:
        # Retrieve container of blocks (e.g.​ transformer layers)
        blocks_container: nn.ModuleDict | nn.ModuleList = attrgetter(attr)(model)
        assert isinstance(blocks_container, nn.ModuleDict | nn.ModuleList), (
            f"model.{attr} must be a nn.ModuleDict or nn.ModuleList, but got {type(blocks_container)}"
        )

        logger.info(
            f"Applying activation checkpointing to {attr} of {type(model)} with ac_freq {ac_freq}; number of blocks: {len(blocks_container)}"
        )

        # Wrap each block with activation-checkpointing logic and re-register it in the container.
        if isinstance(blocks_container, nn.ModuleDict):
            for layer_id, block in blocks_container.items():
                block = _apply_ac_to_block(block, ac_freq)
                blocks_container.register_module(layer_id, block)
        else:  # nn.ModuleList
            for index, block in enumerate(blocks_container):
                block = _apply_ac_to_block(block, ac_freq)
                blocks_container[index] = block


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward_policy: str = "default",
    cast_forward_inputs: bool = False,
    blocks_attr: str | list[str] = "blocks",
    blocks_per_shard_group: int = 1,
) -> None:
    """
    Modified from: https://github.com/pytorch/torchtitan/blob/7d5f3cc698853d2227cf5433776406d0e0345424/torchtitan/models/llama3/infra/parallelize.py#L324

    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.
        cast_forward_inputs (bool, optional): Whether to cast forward inputs to the reduce_dtype. Defaults to False.
        blocks_attr (str, optional): Name of the attribute containing the model blocks (e.g., "blocks", "layers"). Defaults to "blocks".
        blocks_per_shard_group (int, optional): Number of consecutive blocks to shard together in each fully_shard call.
            Defaults to 1 (shard each block individually). Values > 1 will group consecutive blocks together.
            Use -1 to skip individual block sharding and shard the entire model at once.

    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=cast_forward_inputs,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # Validate blocks_per_shard_group parameter
    if blocks_per_shard_group < -1 or blocks_per_shard_group == 0:
        raise ValueError(
            f"blocks_per_shard_group must be >= 1 or -1 (for whole model sharding), got {blocks_per_shard_group}"
        )

    # Retrieve container of blocks (e.g.​ transformer layers)
    if isinstance(blocks_attr, str):
        blocks_attr = [blocks_attr]

    block_list = []
    for attr in blocks_attr:
        blocks_container: nn.ModuleDict | nn.ModuleList = attrgetter(attr)(model)
        assert isinstance(blocks_container, nn.ModuleDict | nn.ModuleList), (
            f"model.{blocks_attr} must be a nn.ModuleDict or nn.ModuleList, but got {type(blocks_container)}"
        )
        logger.info(
            f"Applying FSDP to {attr} of {type(model)}, number of blocks: {len(blocks_container)}, "
            f"blocks_per_shard_group={blocks_per_shard_group}, "
            f"reshard_after_forward_policy={reshard_after_forward_policy}, "
            f"mesh={dp_mesh.mesh_dim_names}, mesh_shape={dp_mesh.shape}, "
            f"mp_policy={mp_policy}"
        )

        block_iterator = blocks_container.values() if isinstance(blocks_container, nn.ModuleDict) else blocks_container
        block_list.extend(list(block_iterator))

    # Handle different sharding strategies
    total_blocks = len(block_list)

    if blocks_per_shard_group == -1:
        # Special case: shard the entire model at once (skip individual block sharding)
        logger.info(f"Sharding entire model at once (skipping individual block sharding for {total_blocks} blocks)")

        # Determine reshard_after_forward for the whole model
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            # For whole model sharding, use default behavior
            reshard_after_forward = False
        else:
            raise ValueError(f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}.")
    else:
        # Standard case: group blocks and apply FSDP to each group
        logger.info(f"Grouping {total_blocks} blocks into groups of {blocks_per_shard_group}")

        # In default mode, we don't reshard the last group and root modules after forward pass
        for group_start in range(0, total_blocks, blocks_per_shard_group):
            group_end = min(group_start + blocks_per_shard_group, total_blocks)
            block_group = block_list[group_start:group_end]

            # Determine reshard_after_forward for this group
            if reshard_after_forward_policy == "always":
                reshard_after_forward = True
            elif reshard_after_forward_policy == "never":
                reshard_after_forward = False
            elif reshard_after_forward_policy == "default":
                # Don't reshard the last group after forward pass
                reshard_after_forward = group_end < total_blocks
            else:
                raise ValueError(f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}.")

            # Apply FSDP to each block in the group
            for block in block_group:
                fully_shard(
                    block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )  # type: ignore

    # Apply FSDP to the root model
    fully_shard(model, **fsdp_config, reshard_after_forward=reshard_after_forward)  # type: ignore

    # Hacky: store fsdp_config in the model
    model.fsdp_config = fsdp_config

    # if reshard_after_forward_policy == "always":
    #     from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule

    #     def _post_fwd(mod, *_):
    #         if isinstance(mod, FSDPModule):
    #             mod.reshard()

    #     model.register_forward_hook(_post_fwd)


def dist_model_setup(
    model: nn.Module,
    shard_size: int | None = None,
    param_dtype: str = "bf16",
    reduce_dtype: str = "fp32",
    ac_freq: int = 0,
    blocks_attr: str | list[str] = "blocks",
    reshard_after_forward_policy: str = "default",
    blocks_per_shard_group: int = 1,
) -> nn.Module:
    """
    Set up a PyTorch model for distributed training with FSDP2 and activation checkpointing.

    This function configures a model for efficient distributed training by applying:
    1. Selective activation checkpointing to reduce memory usage
    2. Fully Sharded Data Parallel v2 (FSDP2) for parameter sharding
    3. Mixed precision training with configurable dtypes
    4. Proper device placement and weight initialization

    The function supports models on 'meta' device (for large model initialization) and
    automatically moves them to CUDA with proper weight initialization.

    Args:
        model: PyTorch model to configure for distributed training. Must be on 'meta' or 'cuda' device.
        shard_size: Number of processes to shard parameters across. If None, uses world_size
                   (full sharding). Must be a divisor of world_size. Defaults to None.
        param_dtype: Data type for model parameters. Supported: "fp32", "fp16", "bf16".
                    Defaults to "bf16".
        reduce_dtype: Data type for gradient reductions. Supported: "fp32", "fp16", "bf16".
                     Defaults to "fp32".
        ac_freq: Activation checkpointing frequency. If > 0, applies checkpointing every
                ac_freq transformer blocks to save memory. If 0, no checkpointing. Defaults to 0.
        blocks_attr: Name of the attribute containing the model blocks (e.g., "blocks", "layers").
                    Defaults to "blocks".
        reshard_after_forward_policy: The policy to use for resharding after forward pass. Defaults to "default".
                    Other options: "never", "always". Use "always" for models without backpropagation (EMA, frozen).
        blocks_per_shard_group: Number of consecutive blocks to shard together in each fully_shard call.
                    Defaults to 1 (shard each block individually). Values > 1 will group consecutive blocks together.
                    Use -1 to skip individual block sharding and shard the entire model at once.

    Returns:
        nn.Module: The configured model ready for distributed training with FSDP2,
                  activation checkpointing, and proper device placement.

    Raises:
        AssertionError: If model is not on 'meta' or 'cuda' device.
        AssertionError: If world_size is not divisible by shard_size.

    Example:
        >>> # Setup model with full sharding and activation checkpointing
        >>> model = MyTransformer().to('meta')
        >>> dist_model = dist_model_setup(
        ...     model,
        ...     shard_size=None,  # Full sharding across all processes
        ...     param_dtype="bf16",
        ...     reduce_dtype="fp32",
        ...     ac_freq=2,  # Checkpoint every 2 blocks
        ...     blocks_attr="blocks"  # Default blocks attribute
        ... )

        >>> # Setup with custom blocks attribute (e.g., LLaMA-style)
        >>> dist_model = dist_model_setup(
        ...     model,
        ...     shard_size=4,  # Shard across 4 processes
        ...     blocks_attr="layers"  # LLaMA-style layers
        ... )

        >>> # Setup with grouped block sharding (shard 2 blocks together)
        >>> dist_model = dist_model_setup(
        ...     model,
        ...     blocks_per_shard_group=2,  # Group 2 consecutive blocks together
        ...     param_dtype="bf16",
        ...     reduce_dtype="fp32"
        ... )

        >>> # Setup with whole model sharding (skip individual block sharding)
        >>> dist_model = dist_model_setup(
        ...     model,
        ...     blocks_per_shard_group=-1,  # Shard entire model at once
        ...     param_dtype="bf16",
        ...     reduce_dtype="fp32"
        ... )

    Note:
        - For large models, start with model on 'meta' device to avoid OOM during initialization
        - Choose shard_size based on your cluster topology for optimal communication
        - Higher ac_freq saves more memory but increases computation time
        - bf16 parameters with fp32 reductions is recommended for stability and performance
        - Different models may have blocks under different attributes (blocks, layers, etc.)
        - blocks_per_shard_group > 1 groups consecutive blocks together for sharding, which can be
          more efficient for communication patterns but may use more memory per FSDP unit
        - blocks_per_shard_group = -1 skips individual block sharding and shards the entire model
          at once, which can be useful for smaller models or specific memory optimization scenarios
    """
    model_device_type = next(model.parameters()).device.type
    if model_device_type == "cpu":
        model = model.cuda()

    # apply selective activation checkpointing ###
    apply_ac(model, ac_freq, blocks_attr)

    # apply FSDP2 ###
    if shard_size is None:
        shard_size = dist.get_world_size()

    assert dist.get_world_size() % shard_size == 0, (
        f"world_size {dist.get_world_size()} must be divisible by shard_size {shard_size}"
    )
    dp_mesh = init_device_mesh(
        "cuda",
        (dist.get_world_size() // shard_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )
    apply_fsdp(
        model,
        dp_mesh,
        DTYPE_MAP[param_dtype],
        DTYPE_MAP[reduce_dtype],
        blocks_attr=blocks_attr,
        reshard_after_forward_policy=reshard_after_forward_policy,
        blocks_per_shard_group=blocks_per_shard_group,
    )

    # init weights ###
    if model_device_type == "meta":
        model.to_empty(device="cuda")

    return model


@contextmanager
def fwd_only_mode(model: nn.Module):
    """Context manager for running validation with proper model state management.

    Handles:
    - Setting model to eval mode (only if needed)
    - FSDP resharding after validation (when no backward pass occurs)
    - Restoring model to train mode (only if it was originally training)
    """
    was_training = model.training
    if was_training:
        model.eval()
    try:
        yield
    finally:
        # FSDP requires manual resharding after validation since there's no backward pass
        if isinstance(model, FSDPModule):
            model.reshard()
        if was_training:
            model.train()