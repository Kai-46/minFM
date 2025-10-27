import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupCosineDecayScheduler:
    """
    Learning rate scheduler with linear warmup, cosine decay, and fixed minimum learning rate.

    Supports per-parameter-group learning rates where each group's learning rate is
    automatically inferred from the optimizer. All groups follow the same warmup and
    decay schedule pattern, decaying to min_lr_ratio * their respective max_lr.
    """

    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
        """
        Initialize the learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer with parameter groups already configured with their learning rates
            warmup_steps: Number of steps for linear warmup (can be 0)
            total_steps: Total number of training steps for warmup + cosine decay
            min_lr_ratio: Ratio to compute min_lr as fraction of each group's max_lr
                         (default: 0.0, meaning decay to 0; 0.1 means decay to 10% of max_lr)
        """
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if total_steps <= warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0.0 and 1.0")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = total_steps - warmup_steps
        self.min_lr_ratio = min_lr_ratio

        # Extract max learning rates from optimizer's parameter groups
        self.max_lrs = [group["lr"] for group in optimizer.param_groups]

        # Compute min learning rates as ratio of max learning rates
        self.min_lrs = [max_lr * min_lr_ratio for max_lr in self.max_lrs]

        # Create lambda functions for each parameter group
        lr_lambdas = []
        for group_max_lr, group_min_lr in zip(self.max_lrs, self.min_lrs):
            lr_lambdas.append(self._make_lr_lambda(group_max_lr, group_min_lr))

        # Create LambdaLR scheduler with per-group lambdas
        self.scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)

    def _make_lr_lambda(self, group_max_lr: float, group_min_lr: float):
        """
        Create a lambda function for a specific parameter group.

        The lambda function returns a multiplicative factor that is applied to the
        base learning rate (group_max_lr) stored in the optimizer.
        """

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                # Linear warmup from near-zero to 1.0
                if self.warmup_steps > 0:
                    return max(1e-8, step / self.warmup_steps)
                else:
                    return 1.0
            elif step < self.total_steps:
                # Cosine decay from 1.0 to min_lr/max_lr ratio
                progress = (step - self.warmup_steps) / self.decay_steps
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                min_ratio = group_min_lr / group_max_lr if group_max_lr > 0 else 0.0
                return min_ratio + (1.0 - min_ratio) * cosine_decay
            else:
                # Fixed at min_lr/max_lr ratio after total_steps
                return group_min_lr / group_max_lr if group_max_lr > 0 else 0.0

        return lr_lambda

    def step(self) -> None:
        """Update learning rate for the next step."""
        self.scheduler.step()

    def get_last_lr(self) -> list:
        """Get the last computed learning rate."""
        return self.scheduler.get_last_lr()

    def state_dict(self) -> dict:
        """Return scheduler state."""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict)