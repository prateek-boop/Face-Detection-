"""
Cosine annealing scheduler with linear warmup + early stopping.

Cold-start training on a random network causes gradient explosions —
warmup prevents this.  Early stopping watches validation mAP and halts
when the model plateaus.
"""

import math


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_epochs : int
    max_epochs : int
    base_lr : float
    min_lr : float
    """

    def __init__(self, optimizer, warmup_epochs: int = 5,
                 max_epochs: int = 100, base_lr: float = 1e-3,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self, epoch: int = None):
        """Update learning rate for the given epoch."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self) -> float:
        """Compute LR for current epoch."""
        e = self.current_epoch

        if e < self.warmup_epochs:
            # Linear warmup: start from min_lr → base_lr
            warmup_factor = (e + 1) / self.warmup_epochs
            return self.min_lr + (self.base_lr - self.min_lr) * warmup_factor
        else:
            # Cosine annealing
            progress = min(1.0, max(0.0, (e - self.warmup_epochs) / max(
                1, self.max_epochs - self.warmup_epochs
            )))
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def state_dict(self):
        return {
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.warmup_epochs,
            "max_epochs": self.max_epochs,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state):
        self.current_epoch = state["current_epoch"]
        self.warmup_epochs = state["warmup_epochs"]
        self.max_epochs = state["max_epochs"]
        self.base_lr = state["base_lr"]
        self.min_lr = state["min_lr"]


class EarlyStopping:
    """Stop training if validation metric doesn't improve for `patience` epochs.

    Parameters
    ----------
    patience : int
        Number of epochs to wait.
    min_delta : float
        Minimum improvement to count as progress.
    mode : str
        ``'max'`` (for mAP) or ``'min'`` (for loss).
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001,
                 mode: str = "max", min_epochs: int = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, metric: float, epoch: int = None) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = metric
            return False

        improved = (metric > self.best_score + self.min_delta) if self.mode == "max" else (metric < self.best_score - self.min_delta)

        if epoch is not None and epoch < self.min_epochs:
            # Do not start early stopping before warmup epochs
            if improved:
                self.best_score = metric
                self.counter = 0
            return False

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience
        return self.should_stop

    def state_dict(self):
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state):
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.should_stop = state["should_stop"]
