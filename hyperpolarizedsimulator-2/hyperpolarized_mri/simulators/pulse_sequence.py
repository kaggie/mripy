import torch

class FlipAngleSimulator:
    """
    Simulate effect of repeated RF pulses (e.g., for SPGR sequence) on magnetization.
    """
    def __init__(self, flip_angles_deg):
        self.flip_angles_rad = torch.tensor(flip_angles_deg) * torch.pi / 180

    def apply_pulses(self, signal):
        # signal: (N_time, N_pools)
        out = signal.clone()
        for idx, angle in enumerate(self.flip_angles_rad):
            out[idx] = out[idx] * torch.cos(angle)
        return out