"""
Optimizer utilities for training.
"""

from dataclasses import dataclass
import fnmatch

from dion import Muon
import torch
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class MatchParamResult:
    matched_param_names: list[str]
    matched_params: list[nn.Parameter]
    unmatched_param_names: list[str]
    unmatched_params: list[nn.Parameter]

    def summarize(self):
        """
        Print a detailed summary table of all matched and unmatched parameters.
        """

        def format_shape(shape: torch.Size) -> str:
            return "x".join(str(d) for d in shape) if len(shape) > 0 else "scalar"

        def format_number(num: int) -> str:
            """Format large numbers with commas"""
            return f"{num:,}"

        def build_param_table(params: list[nn.Parameter], names: list[str], title: str) -> tuple[str, int]:
            """Build a formatted table of parameters as a string"""
            lines = []

            if not params:
                lines.append(f"\n{title}: None")
                return "\n".join(lines), 0

            # Calculate total elements
            total_elements = sum(p.numel() for p in params)

            # Find the maximum name length for proper alignment
            max_name_len = max(len(name) for name in names) if names else 0
            max_name_len = max(max_name_len, len("Parameter Name"))  # At least as wide as the header

            lines.append(f"\n{title} ({len(params)} parameters, {format_number(total_elements)} total elements):")
            lines.append("-" * (max_name_len + 65))

            # Header
            header = f"{'Parameter Name':<{max_name_len}} | {'Shape':^15} | {'Elements':^12} | {'Dtype':^10} | {'Device':^10}"
            lines.append(header)
            lines.append("-" * (max_name_len + 65))

            # Build each parameter row
            for name, param in zip(names, params, strict=False):
                shape_str = format_shape(param.shape)
                elements_str = format_number(param.numel())
                dtype_str = str(param.dtype).replace("torch.", "")
                device_str = str(param.device)

                row = f"{name:<{max_name_len}} | {shape_str:^15} | {elements_str:^12} | {dtype_str:^10} | {device_str:^10}"
                lines.append(row)

            lines.append("-" * (max_name_len + 65))
            return "\n".join(lines), total_elements

        # Build the entire summary as a string
        summary_lines = []

        # Summary header
        summary_lines.append("\n" + "=" * 120)
        summary_lines.append("PARAMETER MATCHING SUMMARY")
        summary_lines.append("=" * 120)

        # Build matched parameters table
        matched_table, matched_elements = build_param_table(
            self.matched_params, self.matched_param_names, "MATCHED PARAMETERS"
        )
        summary_lines.append(matched_table)

        # Build unmatched parameters table
        unmatched_table, unmatched_elements = build_param_table(
            self.unmatched_params, self.unmatched_param_names, "UNMATCHED PARAMETERS"
        )
        summary_lines.append(unmatched_table)

        # Overall summary
        total_params = len(self.matched_params) + len(self.unmatched_params)
        total_elements = matched_elements + unmatched_elements

        summary_lines.append("\n" + "=" * 120)
        summary_lines.append("OVERALL SUMMARY")
        summary_lines.append("=" * 120)

        if total_params > 0:
            matched_param_pct = len(self.matched_params) / total_params * 100
            unmatched_param_pct = len(self.unmatched_params) / total_params * 100

            if total_elements > 0:
                matched_elem_pct = matched_elements / total_elements * 100
                unmatched_elem_pct = unmatched_elements / total_elements * 100
            else:
                matched_elem_pct = unmatched_elem_pct = 0

            summary_data = [
                [
                    "Matched",
                    f"{len(self.matched_params)}",
                    f"{matched_param_pct:.1f}%",
                    f"{format_number(matched_elements)}",
                    f"{matched_elem_pct:.1f}%",
                ],
                [
                    "Unmatched",
                    f"{len(self.unmatched_params)}",
                    f"{unmatched_param_pct:.1f}%",
                    f"{format_number(unmatched_elements)}",
                    f"{unmatched_elem_pct:.1f}%",
                ],
                ["Total", f"{total_params}", "100.0%", f"{format_number(total_elements)}", "100.0%"],
            ]

            # Build summary table
            summary_lines.append(
                f"{'Category':<20} | {'# Parameters':^15} | {'% Parameters':^15} | {'# Elements':^15} | {'% Elements':^15}"
            )
            summary_lines.append("-" * 85)
            for row in summary_data:
                summary_lines.append(f"{row[0]:<20} | {row[1]:^15} | {row[2]:^15} | {row[3]:^15} | {row[4]:^15}")
            summary_lines.append("=" * 120)
        else:
            summary_lines.append("No parameters found!")

        # Single logger call with the complete summary
        logger.info("\n".join(summary_lines))


def match_param_patterns(
    model: nn.Module, param_patterns: list[str] | None = None, include_frozen: bool = False
) -> MatchParamResult:
    """
    Example:
    param_patterns = [
        # Attention layers in double blocks (weight matrices only)
        "double_blocks.*.txt_attn.qkv.weight",
        "double_blocks.*.txt_attn.proj.weight",
        "double_blocks.*.img_attn.qkv.weight",
        "double_blocks.*.img_attn.proj.weight",
        # MLP layers in double blocks (weight matrices only)
        "double_blocks.*.txt_mlp.*.weight",
        "double_blocks.*.img_mlp.*.weight",
        # Attention layers in single blocks (weight matrices only)
        "single_blocks.*.linear1.weight",  # Contains qkv and mlp_in
        "single_blocks.*.linear2.weight",  # Contains proj and mlp_out
        # Note: Excludes txt_in, img_in, final_layer.linear (input/output projections)
        # Note: Excludes all biases by explicitly matching only .weight
    ]
    """
    matched_params = []
    matched_param_names = []
    unmatched_params = []
    unmatched_param_names = []

    # Iterate through all named parameters
    for name, param in model.named_parameters():
        if not param.requires_grad and not include_frozen:
            continue

        # Check if this parameter matches any Muon pattern
        matched = False

        for pattern in param_patterns:
            # Support wildcards in patterns
            if fnmatch.fnmatch(name, pattern):
                matched = True
                break

        if matched:
            matched_params.append(param)
            matched_param_names.append(name)
        else:
            unmatched_params.append(param)
            unmatched_param_names.append(name)

    return MatchParamResult(
        matched_param_names=matched_param_names,
        matched_params=matched_params,
        unmatched_param_names=unmatched_param_names,
        unmatched_params=unmatched_params,
    )


def create_muon_optimizer(
    model: nn.Module,
    adam_lr: float,
    adam_betas: tuple[float, float],
    muon_param_patterns: list[str],
    muon_lr: float,
    muon_mu: float,
    muon_adjust_lr: str,
    weight_decay: float,
) -> list[Optimizer]:
    """
    Create Muon optimizer for the model with separate learning rates for Muon and AdamW parameters.
    """
    # https://github.com/pytorch/pytorch/blob/6c05ea6475beaf3acc05e1bda0f3f8fe3bdc1d49/torch/distributed/fsdp/_fully_shard/_fsdp_common.py#L52
    assert isinstance(model, FSDPModule), f"Model must be an FSDPModule, but got {type(model)}"
    assert hasattr(model, "fsdp_config"), "Model must have fsdp_config"
    shard_mesh = model.fsdp_config["mesh"]["shard"]

    match_param_result = match_param_patterns(model, muon_param_patterns)
    match_param_result.summarize()

    muon_params = match_param_result.matched_params
    muon_param_groups = [
        {
            "params": muon_params,
            "algorithm": "muon",
            "lr": muon_lr,
            "mu": muon_mu,
            "adjust_lr": muon_adjust_lr,
            "weight_decay": weight_decay,
        },
    ]

    adam_params = match_param_result.unmatched_params
    if weight_decay > 0:
        adam_params_decay, adam_params_no_decay = [], []
        for param in adam_params:
            if param.ndim > 1:
                adam_params_decay.append(param)
            else:
                adam_params_no_decay.append(param)
        adam_param_groups = [
            {
                "params": adam_params_decay,
                "algorithm": "adamw",
                "lr": adam_lr,
                "betas": adam_betas,
                "weight_decay": weight_decay,
            },
            {
                "params": adam_params_no_decay,
                "algorithm": "adamw",
                "lr": adam_lr,
                "betas": adam_betas,
                "weight_decay": 0.0,
            },
        ]
    else:
        adam_param_groups = [
            {
                "params": adam_params,
                "algorithm": "adamw",
                "lr": adam_lr,
                "betas": adam_betas,
                "weight_decay": weight_decay,
            },
        ]

    param_groups = muon_param_groups + adam_param_groups

    muon_optimizer = Muon(param_groups, distributed_mesh=shard_mesh, nesterov=True, use_triton=True)

    return muon_optimizer


def create_adam_optimizer(model: nn.Module, lr: float, betas: tuple[float, float], weight_decay: float) -> Optimizer:
    """
    Create AdamW optimizer for the model.
    """
    if weight_decay > 0:
        adam_param_decay, adam_param_no_decay = [], []
        for param in model.parameters():
            if param.ndim > 1:
                adam_param_decay.append(param)
            else:
                adam_param_no_decay.append(param)
        param_groups = [
            {"params": adam_param_decay, "lr": lr, "betas": betas, "weight_decay": weight_decay},
            {"params": adam_param_no_decay, "lr": lr, "betas": betas, "weight_decay": 0.0},
        ]
    else:
        param_groups = [
            {"params": model.parameters(), "lr": lr, "betas": betas, "weight_decay": weight_decay},
        ]

    return AdamW(param_groups, fused=True)


def create_optimizer(
    model: nn.Module,
    use_muon: bool,
    adam_lr: float,
    adam_betas: tuple[float, float],
    muon_param_patterns: list[str],
    muon_lr: float,
    muon_mu: float,
    muon_adjust_lr: str,
    weight_decay: float,
) -> Optimizer:
    if use_muon:
        return create_muon_optimizer(
            model, adam_lr, adam_betas, muon_param_patterns, muon_lr, muon_mu, muon_adjust_lr, weight_decay
        )
    else:
        return create_adam_optimizer(model, adam_lr, adam_betas, weight_decay)