import torch
import torch.nn as nn

class KineticModelBase(nn.Module):
    """
    Abstract base class for kinetic models.
    """
    def __init__(self):
        super().__init__()

    def forward(self, time, params):
        """
        Simulate signals given time points and model parameters.
        Args:
            time (torch.Tensor): 1D tensor of time points
            params (dict): Model parameters
        Returns:
            torch.Tensor: Simulated signal
        """
        raise NotImplementedError