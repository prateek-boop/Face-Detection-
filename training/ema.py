import torch
import torch.nn as nn
from copy import deepcopy

class ModelEMA:
    """ 
    Exponential Moving Average of model weights.
    Keeps a 'shadow' copy of model parameters that updates slowly.
    EMA weights are typically more robust and yield higher mAP.
    """
    def __init__(self, model, decay=0.9999, updates=0):
        # Create shadow copy of the model
        self.ema = deepcopy(model).eval()
        self.updates = updates
        # Static decay for simplicity, or slightly ramp up
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update EMA parameters."""
        with torch.no_grad():
            self.updates += 1
            # Ramping up decay from 0.9 to target decay over first 1000 steps
            d = min(self.decay, (1 + self.updates) / (10 + self.updates) * self.decay)

            # Update all parameters (weights/biases)
            for ema_p, p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(d).add_(p.data, alpha=1 - d)
            
            # Update all buffers (e.g., BN running mean/var)
            for ema_b, b in zip(self.ema.buffers(), model.buffers()):
                ema_b.data.copy_(b.data)

    def state_dict(self):
        return {
            "model": self.ema.state_dict(),
            "updates": self.updates
        }

    def load_state_dict(self, state):
        self.ema.load_state_dict(state["model"])
        self.updates = state.get("updates", 0)
