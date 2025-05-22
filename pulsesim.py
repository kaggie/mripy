import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple

##bir4 = BIR4Pulse(duration=0.01, num_points=256, bw=4000, beta=4, phi=np.pi)
##real, imag = bir4()

class BIR4Pulse(nn.Module):
    """Design a BIR-4 adiabatic pulse."""
    def __init__(self, duration: float, num_points: int, bw: float, beta: float, phi: float):
        """
        Args:
            duration: Pulse duration in seconds.
            num_points: Number of discrete points in the pulse.
            bw: Bandwidth of the pulse (Hz).
            beta: Adiabaticity parameter.
            phi: Phase sweep (radians).
        """
        super().__init__()
        self.duration = duration
        self.num_points = num_points
        self.bw = bw
        self.beta = beta
        self.phi = phi

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.linspace(0, self.duration, self.num_points) - self.duration/2
        norm_t = t / (self.duration/2)
        phi1 = self.phi * norm_t
        amp = torch.tanh(self.beta * norm_t)
        freq = self.bw * torch.tanh(self.beta * norm_t)
        rf = amp * torch.exp(1j * phi1)
        # BIR-4: 4 segments, alternating phase
        rf = torch.cat([
            rf,
            torch.flip(rf, [0]) * torch.exp(1j * self.phi),
            -rf,
            -torch.flip(rf, [0]) * torch.exp(1j * self.phi)
        ])
        return rf.real, rf.imag

class HypSecPulse(nn.Module):
    """Design a hyperbolic secant adiabatic pulse."""
    def __init__(self, duration: float, num_points: int, beta: float, mu: float):
        """
        Args:
            duration: Pulse duration in seconds.
            num_points: Number of discrete points in the pulse.
            beta: Frequency sweep parameter.
            mu: Amplitude scaling parameter.
        """
        super().__init__()
        self.duration = duration
        self.num_points = num_points
        self.beta = beta
        self.mu = mu

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.linspace(-self.duration/2, self.duration/2, self.num_points)
        amp = torch.tensor([1/np.cosh(self.beta * float(tt)) for tt in t])
        freq = self.mu * torch.tanh(self.beta * t)
        phase = 2 * np.pi * torch.cumsum(freq, dim=0) * (self.duration/self.num_points)
        rf = amp * torch.exp(1j*phase)
        return rf.real, rf.imag

class WURSTPulse(nn.Module):
    """Design a WURST adiabatic pulse."""
    def __init__(self, duration: float, num_points: int, n: float, bw: float, phi: float = 0):
        """
        Args:
            duration: Pulse duration in seconds.
            num_points: Number of discrete points.
            n: WURST truncation parameter (controls smoothness).
            bw: Frequency sweep (Hz).
            phi: Phase offset.
        """
        super().__init__()
        self.duration = duration
        self.num_points = num_points
        self.n = n
        self.bw = bw
        self.phi = phi

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.linspace(-self.duration/2, self.duration/2, self.num_points)
        amp = 1 - torch.abs(torch.sin(np.pi * t / self.duration)) ** self.n
        freq = self.bw * t / self.duration
        phase = 2 * np.pi * torch.cumsum(freq, dim=0) * (self.duration/self.num_points) + self.phi
        rf = amp * torch.exp(1j * phase)
        return rf.real, rf.imag

class GOIAWURSTPulse(nn.Module):
    """Design a GOIA WURST adiabatic pulse."""
    def __init__(self, duration: float, num_points: int, n: float, bw: float, gmax: float, phi: float = 0):
        """
        Args:
            duration: Pulse duration (s).
            num_points: Number of points.
            n: WURST truncation parameter.
            bw: Frequency sweep (Hz).
            gmax: Maximum gradient value.
            phi: Phase offset.
        """
        super().__init__()
        self.duration = duration
        self.num_points = num_points
        self.n = n
        self.bw = bw
        self.gmax = gmax
        self.phi = phi

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.linspace(-self.duration/2, self.duration/2, self.num_points)
        amp = 1 - torch.abs(torch.sin(np.pi * t / self.duration)) ** self.n
        freq = self.bw * t / self.duration
        grad = self.gmax * torch.sin(np.pi * t / self.duration)
        phase = 2 * np.pi * torch.cumsum(freq, dim=0) * (self.duration/self.num_points) + self.phi
        rf = amp * torch.exp(1j * phase)
        return rf.real, rf.imag, grad

class BlochSiegertFMPulse(nn.Module):
    """U-shaped FM waveform for adiabatic Bloch-Siegert B1+ mapping and spatial encoding."""
    def __init__(self, duration: float, num_points: int, bw: float, phi: float = 0):
        """
        Args:
            duration: Pulse duration (s).
            num_points: Number of points.
            bw: Frequency sweep (Hz).
            phi: Phase offset.
        """
        super().__init__()
        self.duration = duration
        self.num_points = num_points
        self.bw = bw
        self.phi = phi

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.linspace(-self.duration/2, self.duration/2, self.num_points)
        # U-shaped FM: f(t) ~ (t/T)^2
        freq = self.bw * (t / (self.duration/2)) ** 2
        amp = torch.ones_like(t)
        phase = 2 * np.pi * torch.cumsum(freq, dim=0) * (self.duration/self.num_points) + self.phi
        rf = amp * torch.exp(1j * phase)
        return rf.real, rf.imag


class HardPulse(nn.Module):
    """Rectangular (hard) RF pulse."""
    def __init__(self, duration: float, num_points: int, flip_angle_deg: float = 90.0, phase: float = 0):
        """
        Args:
            duration: Pulse duration (s).
            num_points: Number of time points.
            flip_angle_deg: Flip angle in degrees (nominal).
            phase: Phase offset in radians.
        """
        super().__init__()
        self.duration = duration
        self.num_points = num_points
        self.flip_angle_deg = flip_angle_deg
        self.phase = phase

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        amp = torch.ones(self.num_points)
        # Normalize for nominal flip angle (gamma*B1*duration = flip_angle)
        gamma = 267.513e6  # rad/T/s
        dt = self.duration / self.num_points
        area = amp.sum() * dt
        b1 = np.deg2rad(self.flip_angle_deg) / (gamma * area)
        rf = amp * b1 * torch.exp(1j * self.phase)
        return rf.real, rf.imag

class SincPulse(nn.Module):
    """Sinc-shaped, optionally windowed, slice-selective RF pulse."""
    def __init__(self, duration: float, num_points: int, time_bw_product: float = 4, 
                 flip_angle_deg: float = 90.0, window: Optional[str] = None, phase: float = 0):
        """
        Args:
            duration: Pulse duration (s).
            num_points: Number of points.
            time_bw_product: Time-bandwidth product (controls number of lobes).
            flip_angle_deg: Nominal flip angle in degrees.
            window: None, "hamming", or "hanning".
            phase: Phase offset.
        """
        super().__init__()
        self.duration = duration
        self.num_points = num_points
        self.time_bw_product = time_bw_product
        self.flip_angle_deg = flip_angle_deg
        self.window = window
        self.phase = phase

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.linspace(-self.duration/2, self.duration/2, self.num_points)
        bw = self.time_bw_product / self.duration
        x = np.pi * bw * t
        sinc = torch.where(x == 0, torch.ones_like(x), torch.sin(x) / x)
        # Windowing
        if self.window is not None:
            if self.window.lower() == "hamming":
                win = torch.hamming_window(self.num_points)
            elif self.window.lower() == "hanning":
                win = torch.hann_window(self.num_points)
            else:
                raise ValueError(f"Unknown window: {self.window}")
            sinc = sinc * win
        # Normalize for flip angle (integral of B1 over time)
        gamma = 267.513e6  # rad/T/s
        dt = self.duration / self.num_points
        area = sinc.sum() * dt
        b1 = np.deg2rad(self.flip_angle_deg) / (gamma * area)
        rf = sinc * b1 * torch.exp(1j * self.phase)
        return rf.real, rf.imag

class GaussianPulse(nn.Module):
    """Gaussian RF pulse."""
    def __init__(self, duration: float, num_points: int, sigma: float, flip_angle_deg: float = 90.0, phase: float = 0):
        """
        Args:
            duration: Pulse duration (s).
            num_points: Number of points.
            sigma: Standard deviation of Gaussian (as fraction of duration).
            flip_angle_deg: Nominal flip angle in degrees.
            phase: Phase offset.
        """
        super().__init__()
        self.duration = duration
        self.num_points = num_points
        self.sigma = sigma
        self.flip_angle_deg = flip_angle_deg
        self.phase = phase

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.linspace(-self.duration/2, self.duration/2, self.num_points)
        sig = self.sigma * self.duration
        gauss = torch.exp(-0.5 * (t / sig) ** 2)
        # Normalize for flip angle
        gamma = 267.513e6  # rad/T/s
        dt = self.duration / self.num_points
        area = gauss.sum() * dt
        b1 = np.deg2rad(self.flip_angle_deg) / (gamma * area)
        rf = gauss * b1 * torch.exp(1j * self.phase)
        return rf.real, rf.imag

class CompositePulse(nn.Module):
    """Composite pulse sequence (e.g., BB1, MLEV)."""
    def __init__(self, pulses: List[nn.Module]):
        """
        Args:
            pulses: List of pulse modules (e.g., HardPulse with different phases).
        """
        super().__init__()
        self.pulses = nn.ModuleList(pulses)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        real_parts = []
        imag_parts = []
        for pulse in self.pulses:
            r, i = pulse()
            real_parts.append(r)
            imag_parts.append(i)
        return torch.cat(real_parts), torch.cat(imag_parts)

class VERSEPulse(nn.Module):
    """Prototype Variable Rate Selective Excitation (VERSE) pulse."""
    def __init__(self, base_pulse: nn.Module, verse_factor: float = 2.0):
        """
        Args:
            base_pulse: Base RF pulse (e.g., SincPulse).
            verse_factor: Factor to stretch pulse in regions of high amplitude.
        """
        super().__init__()
        self.base_pulse = base_pulse
        self.verse_factor = verse_factor

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_real, base_imag = self.base_pulse()
        amp = torch.sqrt(base_real ** 2 + base_imag ** 2)
        max_amp = amp.max()
        # Stretch where amplitude is high
        time_stretch = 1 + (self.verse_factor - 1) * (amp / max_amp)
        # Interpolate to new timeline
        orig_t = torch.linspace(0, 1, len(amp))
        new_t = torch.cumsum(time_stretch, dim=0)
        new_t = new_t / new_t[-1]
        real = torch.interp(orig_t, new_t, base_real)
        imag = torch.interp(orig_t, new_t, base_imag)
        return real, imag

class MultibandPulse(nn.Module):
    """Multiband (Simultaneous Multi-Slice) RF pulse."""
    def __init__(self, base_pulse: nn.Module, num_bands: int = 2, band_spacing: float = 2000.0):
        """
        Args:
            base_pulse: Base RF pulse (e.g., SincPulse).
            num_bands: Number of simultaneous bands.
            band_spacing: Frequency spacing between bands (Hz).
        """
        super().__init__()
        self.base_pulse = base_pulse
        self.num_bands = num_bands
        self.band_spacing = band_spacing

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_real, base_imag = self.base_pulse()
        n = len(base_real)
        t = torch.linspace(-0.5, 0.5, n)
        rf_sum = torch.zeros(n, dtype=torch.complex64)
        for b in range(self.num_bands):
            freq_shift = (b - (self.num_bands - 1) / 2) * self.band_spacing
            phase = torch.exp(1j * 2 * np.pi * freq_shift * t)
            rf_band = (base_real + 1j * base_imag) * phase
            rf_sum += rf_band
        return rf_sum.real, rf_sum.imag

class SLRPulse(nn.Module):
    """Placeholder for numerically optimized SLR pulse. (Not implemented)"""
    def __init__(self):
        super().__init__()
        # Real SLR design requires numeric optimization (not analytic).
        raise NotImplementedError("SLR pulse design requires numerical optimization (not implemented here).")

# Example: Composite BB1 180°
def bb1_180_sequence(duration: float, num_points: int) -> CompositePulse:
    # 90x - 180y - 90x sequence
    hard90x = HardPulse(duration/3, num_points//3, 90, 0)
    hard180y = HardPulse(duration/3, num_points//3, 180, np.pi/2)
    hard90x2 = HardPulse(duration/3, num_points//3, 90, 0)
    return CompositePulse([hard90x, hard180y, hard90x2])
