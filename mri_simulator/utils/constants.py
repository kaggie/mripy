# mri_simulator/utils/constants.py
import torch

# Gyromagnetic ratio for Hydrogen (1H) in Hz/T
GAMMA_HZ_T_PROTON = 42.576e6

# Gyromagnetic ratio for Hydrogen (1H) in rad/s/T
GAMMA_RAD_S_T_PROTON = GAMMA_HZ_T_PROTON * 2 * torch.pi

# Main magnetic field strength in Tesla (default, can be changed)
B0_TESLA = 1.5

# Print constants for verification
if __name__ == '__main__':
    print(f"Gyromagnetic ratio (1H) [Hz/T]: {GAMMA_HZ_T_PROTON}")
    print(f"Gyromagnetic ratio (1H) [rad/s/T]: {GAMMA_RAD_S_T_PROTON}")
    print(f"Default B0 field strength [T]: {B0_TESLA}")
