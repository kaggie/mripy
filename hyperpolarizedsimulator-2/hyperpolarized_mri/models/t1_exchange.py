import torch
from .base_model import KineticModelBase

class T1RecoveryExchangeModel(KineticModelBase):
    """
    T1 recovery with two-site exchange.
    """
    def forward(self, time, params):
        # params: {'kAB', 'kBA', 'T1A', 'T1B', 'M0A', 'M0B'}
        kAB = params['kAB']
        kBA = params['kBA']
        T1A = params['T1A']
        T1B = params['T1B']
        M0A = params['M0A']
        M0B = params['M0B']

        # Placeholder for recovery + exchange model
        A = M0A * (1 - torch.exp(-time/T1A)) * torch.exp(-kAB * time)
        B = M0B * (1 - torch.exp(-time/T1B)) * torch.exp(-kBA * time)
        return torch.stack([A, B], dim=1)