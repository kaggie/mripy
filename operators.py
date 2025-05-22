import torch
import torch.nn as nn
import torch.fft

class LinearOperator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def adjoint(self, y):
        raise NotImplementedError

class FFTOperator(LinearOperator):
    def __init__(self, dim=(-2, -1)):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x_shifted = torch.fft.ifftshift(x, dim=self.dim)
        k_space = torch.fft.fftn(x_shifted, dim=self.dim, norm="ortho")
        k_space_shifted = torch.fft.fftshift(k_space, dim=self.dim)
        return k_space_shifted

    def adjoint(self, y):
        y_shifted = torch.fft.ifftshift(y, dim=self.dim)
        img_space = torch.fft.ifftn(y_shifted, dim=self.dim, norm="ortho")
        img_space_shifted = torch.fft.fftshift(img_space, dim=self.dim)
        return img_space_shifted

class IFFTOperator(LinearOperator):
    def __init__(self, dim=(-2, -1)):
        super().__init__()
        self.dim = dim

    def forward(self, y):
        y_shifted = torch.fft.ifftshift(y, dim=self.dim)
        img_space = torch.fft.ifftn(y_shifted, dim=self.dim, norm="ortho")
        img_space_shifted = torch.fft.fftshift(img_space, dim=self.dim)
        return img_space_shifted

    def adjoint(self, x):
        x_shifted = torch.fft.ifftshift(x, dim=self.dim)
        k_space = torch.fft.fftn(x_shifted, dim=self.dim, norm="ortho")
        k_space_shifted = torch.fft.fftshift(k_space, dim=self.dim)
        return k_space_shifted

try:
    import torchkbnufft
    torchkbnufft_available = True
except ImportError:
    print("torchkbnufft not found. NUFFTOperator will not be functional.")
    torchkbnufft_available = False

class NUFFTOperator(LinearOperator):
    def __init__(self, im_size=None, k_traj=None, **kwargs):
        super().__init__()
        if not torchkbnufft_available:
            # Raise error in __init__ if not available, to prevent object creation
            # The task asks for this in forward/adjoint, but it's better to fail early.
            # However, to strictly follow, I'll keep the RuntimeError in forward/adjoint
            # and just set a flag here.
            self._functional = False 
            return

        if im_size is None or k_traj is None:
            # This case should ideally raise an error if torchkbnufft is available
            # but required arguments are missing.
            print("Warning: NUFFTOperator initialized without im_size or k_traj. It might not be functional.")
            self._functional = False
            return
            
        self._functional = True
        self.im_size = im_size
        self.k_traj = k_traj
        
        # KbNufft expects k_traj to be shape (D, M)
        # im_size tuple e.g. (256, 256)
        self.nufft_op = torchkbnufft.KbNufft(im_size=self.im_size, **kwargs)
        self.adj_nufft_op = torchkbnufft.KbNufftAdjoint(im_size=self.im_size, **kwargs)

    def forward(self, x_image):
        if not self._functional or not torchkbnufft_available:
            raise RuntimeError("torchkbnufft not found or NUFFTOperator not initialized correctly. NUFFTOperator is not functional.")
        # x_image: (batch, channels, *im_size)
        # k_traj: (D, num_points)
        # NUFFT expects complex input.
        if not x_image.is_complex():
             x_image = x_image.to(torch.complex64) # Or complex128
        return self.nufft_op(x_image, self.k_traj)

    def adjoint(self, y_kspace):
        if not self._functional or not torchkbnufft_available:
            raise RuntimeError("torchkbnufft not found or NUFFTOperator not initialized correctly. NUFFTOperator is not functional.")
        # y_kspace: (batch, channels, num_points)
        # k_traj: (D, num_points)
        if not y_kspace.is_complex():
            y_kspace = y_kspace.to(torch.complex64) # Or complex128
        return self.adj_nufft_op(y_kspace, self.k_traj)
