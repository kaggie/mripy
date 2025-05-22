

import torch  
import torch.nn as nn  
import torch.optim as optim  
import numpy as np

class BaseMRIFitter(nn.Module):  
    """Base class for MRI fitting techniques."""  
    def _*init*_(self, device='cuda' if torch.cuda.is_available() else 'cpu'):  
        super()._*init*_()  
        self.device = device  
        self.to(device)

def fit(self, data, params_init, max_iter=1000, lr=0.01):  
        """Generic fitting method using gradient descent."""  
        params = torch.tensor(params_init, requires_grad=True, device=self.device)  
        optimizer = optim.Adam([params], lr=lr)  
        criterion = nn.MSELoss()

for _ in range(max_iter):  
            optimizer.zero_grad()  
            predicted = self.forward(data['x'])  
            loss = criterion(predicted, data['y'])  
            loss.backward()  
            optimizer.step()  
            if loss.item() < 1e-6:  
                break  
        return params.detach().cpu().numpy()

class DESPOT1Fitter(BaseMRIFitter):  
    """Fitter for DESPOT1 (T1 estimation)."""  
  def _*init*_(self, TR, flip_angles, device='cuda' if torch.cuda.is_available() else 'cpu'):  
        super()._*init*_(device)  
        self.TR = torch.tensor(TR, device=device)  
        self.flip_angles = torch.tensor(flip_angles, device=device)

  def forward(self, params):  
        """DESPOT1 signal model: S = M0 * sin(alpha) * (1 - E1) / (1 - cos(alpha) * E1)."""  
        M0, T1 = params[:, 0], params[:, 1]  
        E1 = torch.exp(-self.TR / T1)  
        signal = M0[:, None] * torch.sin(self.flip_angles) * (1 - E1[:, None]) / (1 - torch.cos(self.flip_angles) * E1[:, None])  
        return signal

  def fit(self, signals, flip_angles, TR, params_init=None):  
        """Fit DESPOT1 model to data."""  
        if params_init is None:  
            params_init = np.ones((signals.shape[0], 2)) # [M0, T1]  
        data = {  
            'x': torch.tensor(flip_angles, device=self.device),  
            'y': torch.tensor(signals, device=self.device)  
        }  
        return super().fit(data, params_init)

class DESPOT2Fitter(BaseMRIFitter):  
    """Fitter for DESPOT2 (T2 estimation with known T1)."""  
    def _*init*_(self, TR, TE, flip_angles, T1, device='cuda' if torch.cuda.is_available() else 'cpu'):  
        super()._*init*_(device)  
        self.TR = torch.tensor(TR, device=device)  
        self.TE = torch.tensor(TE, device=device)  
        self.flip_angles = torch.tensor(flip_angles, device=device)  
        self.T1 = torch.tensor(T1, device=device)

   def forward(self, params):  
        """DESPOT2 signal model (SSFP)."""  
        M0, T2 = params[:, 0], params[:, 1]  
        E1 = torch.exp(-self.TR / self.T1)  
        E2 = torch.exp(-self.TE / T2)  
        signal = M0[:, None] * torch.sin(self.flip_angles) * (1 - E1[:, None]) / (1 - torch.cos(self.flip_angles) * E1[:, None]) * E2[:, None]  
        return signal

class BiexponentialFitter(BaseMRIFitter):  
    """Fitter for biexponential decay (e.g., T2 or diffusion)."""  
  def _*init*_(self, times, device='cuda' if torch.cuda.is_available() else 'cpu'):  
        super()._*init*_(device)  
        self.times = torch.tensor(times, device=device)

  def forward(self, params):  
        """Biexponential model: S(t) = A1 * exp(-t/T2_1) + A2 * exp(-t/T2_2)."""  
        A1, T2_1, A2, T2_2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]  
        signal = A1[:, None] * torch.exp(-self.times / T2_1[:, None]) + A2[:, None] * torch.exp(-self.times / T2_2[:, None])  
        return signal

class T2StarFitter(BaseMRIFitter):  
    """Fitter for T2* exponential decay."""  
    def _*init*_(self, TE, device='cuda' if torch.cuda.is_available() else 'cpu'):  
        super()._*init*_(device)  
        self.TE = torch.tensor(TE, device=device)

  def forward(self, params):  
        """T2* model: S(t) = M0 * exp(-t/T2*)."""  
        M0, T2_star = params[:, 0], params[:, 1]  
        signal = M0[:, None] * torch.exp(-self.TE / T2_star[:, None])  
        return signal

class InversionRecoveryFitter(BaseMRIFitter):  
    """Fitter for inversion recovery sequences."""  
    def _*init*_(self, TI, TR, device='cuda' if torch.cuda.is_available() else 'cpu'):  
        super()._*init*_(device)  
        self.TI = torch.tensor(TI, device=device)  
        self.TR = torch.tensor(TR, device=device)

  def forward(self, params):  
        """IR model: S(TI) = M0 * (1 - 2 * exp(-TI/T1) + exp(-TR/T1))."""  
        M0, T1 = params[:, 0], params[:, 1]  
        signal = M0[:, None] * (1 - 2 * torch.exp(-self.TI / T1[:, None]) + torch.exp(-self.TR / T1[:, None]))  
        return signal

class PhaseUnwrapper:  
    """Custom iterative phase unwrapping algorithm."""  
    def _*init*_(self, device='cuda' if torch.cuda.is_available() else 'cpu'):  
        self.device = device

  def unwrap(self, phase, mask=None, max_iter=100):  
        """Unwrap phase using iterative gradient-based method."""  
        phase = torch.tensor(phase, device=self.device)  
        unwrapped = phase.clone()  
        if mask is None:  
            mask = torch.ones_like(phase, device=self.device)

  for _ in range(max_iter):  
            # Compute wrapped phase differences  
            grad_x = torch.diff(unwrapped, dim=-1, prepend=unwrapped[..., :1])  
            grad_y = torch.diff(unwrapped, dim=-2, prepend=unwrapped[..., :1, :])  
            # Identify wrapping discontinuities  
            wrap_x = torch.where(torch.abs(grad_x) > np.pi, torch.sign(grad_x) * 2 * np.pi, 0)  
            wrap_y = torch.where(torch.abs(grad_y) > np.pi, torch.sign(grad_y) * 2 * np.pi, 0)  
            # Correct phase  
            unwrapped[..., 1:] -= torch.cumsum(wrap_x * mask[..., 1:], dim=-1)  
            unwrapped[..., 1:, :] -= torch.cumsum(wrap_y * mask[..., 1:, :], dim=-2)  
            if torch.all(wrap_x == 0) and torch.all(wrap_y == 0):  
                break  
        return unwrapped.cpu().numpy()

class QSMFitter(BaseMRIFitter):  
    """Fitter for quantitative susceptibility mapping."""  
    def _*init*_(self, field_map, mask, dipole_kernel, device='cuda' if torch.cuda.is_available() else 'cpu'):  
        super()._*init*_(device)  
        self.field_map = torch.tensor(field_map, device=device)  
        self.mask = torch.tensor(mask, device=device)  
        self.dipole_kernel = torch.tensor(dipole_kernel, device=device)

  def forward(self, chi):  
        """Dipole convolution model for QSM."""  
        chi_k = torch.fft.fftn(chi)  
        field = torch.fft.ifftn(chi_k * self.dipole_kernel).real  
        return field * self.mask

  def fit(self, field_map, params_init=None):  
        """Fit susceptibility map."""  
        if params_init is None:  
            params_init = np.zeros_like(field_map)  
        data = {'x': None, 'y': self.field_map}  
        return super().fit(data, params_init)

class DiffusionFitter(BaseMRIFitter):  
    """Fitter for diffusion-weighted imaging (monoexponential or tensor)."""  
    def _*init*_(self, b_values, device='cuda' if torch.cuda.is_available() else 'cpu'):  
        super()._*init*_(device)  
        self.b_values = torch.tensor(b_values, device=device)

  def forward(self, params):  
        """Monoexponential diffusion model: S(b) = S0 * exp(-b * D)."""  
        S0, D = params[:, 0], params[:, 1]  
        signal = S0[:, None] * torch.exp(-self.b_values * D[:, None])  
        return signal

class NonLinear3DMapFitter(BaseMRIFitter):  
    """Fitter for non-linear 3D parameter maps."""  
  def _*init*_(self, model_func, x_data, device='cuda' if torch.cuda.is_available() else 'cpu'):  
        super()._*init*_(device)  
        self.model_func = model_func  
        self.x_data = torch.tensor(x_data, device=device)

  def forward(self, params):  
        """Custom non-linear model for 3D maps."""  
        return self.model_func(self.x_data, params)

# Example usage  
if _name_ == "_main_":  
  # Example data  
    flip_angles = np.array([5, 10, 15, 20]) * np.pi / 180  
    TR = 10.0 # ms  
    signals = np.random.rand(100, len(flip_angles)) # Simulated signals  
    despot1 = DESPOT1Fitter(TR, flip_angles)  
    params = despot1.fit(signals, flip_angles, TR)  
    print("DESPOT1 Parameters (M0, T1):", params)

  # Phase unwrapping example  
    phase = np.random.rand(64, 64) * 2 * np.pi - np.pi  
    unwrapper = PhaseUnwrapper()  
    unwrapped_phase = unwrapper.unwrap(phase)  
    print("Unwrapped phase shape:", unwrapped_phase.shape)
