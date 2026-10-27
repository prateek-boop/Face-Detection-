"""
Checkpoint manager — saves/loads full training state and keeps top-K by mAP.

Saves: model weights, optimizer state, scheduler state, epoch, metrics.
Always keeps the latest checkpoint + top-3 by validation mAP.
"""

import os
import glob
import json
import torch


class CheckpointManager:
    """Manages training checkpoints.

    Parameters
    ----------
    save_dir : str
        Directory for checkpoint files.
    keep_top_k : int
        Number of best checkpoints to keep (ranked by mAP).
    """

    def __init__(self, save_dir: str = "checkpoints", keep_top_k: int = 3):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        os.makedirs(save_dir, exist_ok=True)

        # Track saved checkpoints: list of (path, mAP)
        self._history = []
        self._scan_dir()

    def _scan_dir(self):
        """Find existing checkpoints in directory."""
        pattern = os.path.join(self.save_dir, "epoch_*_mAP_*.pt")
        for p in glob.glob(pattern):
            try:
                # epoch_0065_mAP_0.1986.pt
                name = os.path.basename(p)
                mAP_str = name.split("mAP_")[1].replace(".pt", "")
                self._history.append((p, float(mAP_str)))
            except:
                continue
        # Sort by mAP
        self._history.sort(key=lambda x: x[1], reverse=True)

    def save(self, model, optimizer, scheduler, epoch: int,
             metrics: dict, early_stopping=None, ema=None):
        """Save full training state.

        Parameters
        ----------
        model : nn.Module
        optimizer : torch.optim.Optimizer
        scheduler : WarmupCosineScheduler
        epoch : int
        metrics : dict (must contain 'mAP')
        early_stopping : EarlyStopping or None
        ema : ModelEMA or None
        """
        mAP = metrics.get("mAP", 0.0)
        filename = f"epoch_{epoch:04d}_mAP_{mAP:.4f}.pt"
        filepath = os.path.join(self.save_dir, filename)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        }
        if early_stopping is not None:
            state["early_stopping_state_dict"] = early_stopping.state_dict()

        if ema is not None:
            state["ema_state_dict"] = ema.state_dict()

        torch.save(state, filepath)

        # Also save as "latest.pt" for crash recovery
        latest_path = os.path.join(self.save_dir, "latest.pt")
        torch.save(state, latest_path)

        # Track and prune
        self._history.append((filepath, mAP))
        self._prune()

        print(f"Saved checkpoint: {filename} (mAP={mAP:.4f})")

    def _prune(self):
        """Keep only top-K checkpoints by mAP + latest."""
        if len(self._history) <= self.keep_top_k:
            return

        # Sort by mAP descending
        sorted_history = sorted(self._history, key=lambda x: x[1], reverse=True)
        keep_paths = {p for p, _ in sorted_history[:self.keep_top_k]}

        # Always keep latest.pt (it's a separate copy)
        latest_path = os.path.join(self.save_dir, "latest.pt")
        keep_paths.add(latest_path)

        # Remove old ones
        new_history = []
        for path, mAP in self._history:
            if path in keep_paths:
                new_history.append((path, mAP))
            else:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Pruned checkpoint: {os.path.basename(path)}")

        self._history = new_history

    @staticmethod
    def load(path: str, model, optimizer=None, scheduler=None,
             early_stopping=None, device="cpu", ema=None):
        """Load checkpoint and restore all states.

        Parameters
        ----------
        path : str
            Path to checkpoint file.
        model, optimizer, scheduler, early_stopping, ema : objects to load into

        Returns
        -------
        epoch : int
        metrics : dict
        """
        state = torch.load(path, map_location=device, weights_only=True)

        model.load_state_dict(state["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        if early_stopping is not None and "early_stopping_state_dict" in state:
            early_stopping.load_state_dict(state["early_stopping_state_dict"])

        if ema is not None and "ema_state_dict" in state:
            ema.load_state_dict(state["ema_state_dict"])

        epoch = state.get("epoch", 0)
        metrics = state.get("metrics", {})

        print(f"Loaded checkpoint from epoch {epoch} (mAP={metrics.get('mAP', 'N/A')})")
        return epoch, metrics

    @staticmethod
    def find_latest(ckpt_dir: str) -> str | None:
        """Find the latest checkpoint for auto-resume.

        Returns
        -------
        path : str or None
        """
        latest = os.path.join(ckpt_dir, "latest.pt")
        if os.path.exists(latest):
            return latest

        # Fallback: find highest-epoch file
        pattern = os.path.join(ckpt_dir, "epoch_*.pt")
        files = sorted(glob.glob(pattern))
        return files[-1] if files else None
