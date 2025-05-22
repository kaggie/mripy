# -*- coding: utf-8 -*-
"""
Created on Sat May 17 10:19:55 2025

@author: Josh
"""

import torch
import numpy as np
from scipy.special import i0

# Kaiser-Bessel kernel
def kaiser_bessel_kernel(r, width, beta):
    mask = r < (width / 2)
    z = torch.sqrt(1 - (2 * r[mask] / width)**2)
    kb = torch.zeros_like(r)
    kb[mask] = torch.from_numpy(i0(beta * z.cpu().numpy())).to(r.device) / i0(beta)
    return kb

# Density compensation (simple radius-based for radial trajectory)
def estimate_density_compensation(kx, ky):
    radius = torch.sqrt(kx**2 + ky**2)
    dcf = radius + 1e-3  # avoid zero center
    dcf /= dcf.max()
    return dcf

# NUFFT type 2: Non-uniform k-space data -> image (adjoint)
def nufft2d2_adjoint(kx, ky, kspace_data, image_shape, oversamp=2.0, width=4, beta=13.9085):
    device = kx.device
    Nx, Ny = image_shape
    Nx_oversamp = int(Nx * oversamp)
    Ny_oversamp = int(Ny * oversamp)

    # Scale k-space coords to oversampled grid
    kx_scaled = (kx + 0.5) * Nx_oversamp
    ky_scaled = (ky + 0.5) * Ny_oversamp

    # Density compensation weights
    dcf = estimate_density_compensation(kx, ky).to(device)
    kspace_data = kspace_data * dcf

    # Initialize oversampled grid and weight grid
    grid = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.complex64, device=device)
    weight = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.float32, device=device)

    half_width = width // 2

    # Gridding: interpolate k-space data to Cartesian grid with Kaiser-Bessel kernel
    for dx in range(-half_width, half_width + 1):
        for dy in range(-half_width, half_width + 1):
            x_idx = torch.floor(kx_scaled + dx).long() % Nx_oversamp
            y_idx = torch.floor(ky_scaled + dy).long() % Ny_oversamp

            x_dist = kx_scaled - x_idx.float()
            y_dist = ky_scaled - y_idx.float()
            r = torch.sqrt(x_dist**2 + y_dist**2)
            w = kaiser_bessel_kernel(r, width, beta)

            for i in range(kspace_data.shape[0]):
                grid[x_idx[i], y_idx[i]] += kspace_data[i] * w[i]
                weight[x_idx[i], y_idx[i]] += w[i]

    weight = torch.where(weight == 0, torch.ones_like(weight), weight)
    grid = grid / weight

    # IFFT and crop
    img = torch.fft.ifftshift(grid)
    img = torch.fft.ifft2(img)
    img = torch.fft.fftshift(img)

    start_x = (Nx_oversamp - Nx) // 2
    start_y = (Ny_oversamp - Ny) // 2
    img_cropped = img[start_x:start_x + Nx, start_y:start_y + Ny]

    return img_cropped

# NUFFT type 1: image -> non-uniform k-space (forward NUFFT)
def nufft2d2_forward(kx, ky, image, oversamp=2.0, width=4, beta=13.9085):
    device = kx.device
    Nx, Ny = image.shape
    Nx_oversamp = int(Nx * oversamp)
    Ny_oversamp = int(Ny * oversamp)

    # Zero pad image to oversampled size
    pad_x = (Nx_oversamp - Nx) // 2
    pad_y = (Ny_oversamp - Ny) // 2
    image_padded = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.complex64, device=device)
    image_padded[pad_x:pad_x + Nx, pad_y:pad_y + Ny] = image

    # FFT
    kspace_cart = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image_padded)))

    # Scale k-space coords to oversampled grid
    kx_scaled = (kx + 0.5) * Nx_oversamp
    ky_scaled = (ky + 0.5) * Ny_oversamp

    half_width = width // 2
    kspace_data = torch.zeros(kx.shape[0], dtype=torch.complex64, device=device)
    weight = torch.zeros(kx.shape[0], dtype=torch.float32, device=device)

    # Interpolate grid values at non-uniform k-space locations
    for dx in range(-half_width, half_width + 1):
        for dy in range(-half_width, half_width + 1):
            x_idx = torch.floor(kx_scaled + dx).long() % Nx_oversamp
            y_idx = torch.floor(ky_scaled + dy).long() % Ny_oversamp

            x_dist = kx_scaled - x_idx.float()
            y_dist = ky_scaled - y_idx.float()
            r = torch.sqrt(x_dist**2 + y_dist**2)
            w = kaiser_bessel_kernel(r, width, beta)

            kspace_data += kspace_cart[x_idx, y_idx] * w
            weight += w

    weight = torch.where(weight == 0, torch.ones_like(weight), weight)
    kspace_data /= weight

    return kspace_data

# Iterative reconstruction using conjugate gradient (CG)
def iterative_recon(kx, ky, kspace_data, image_shape, oversamp=2.0, width=4, beta=13.9085, num_iters=10):
    device = kx.device
    Nx, Ny = image_shape

    # Initial guess (zero image)
    x = torch.zeros((Nx, Ny), dtype=torch.complex64, device=device)

    # Helper lambda for forward NUFFT
    def A(img):
        return nufft2d2_forward(kx, ky, img, oversamp, width, beta)

    # Helper lambda for adjoint NUFFT
    def At(data):
        return nufft2d2_adjoint(kx, ky, data, image_shape, oversamp, width, beta)

    # Precompute A^H y
    AHy = At(kspace_data)

    r = AHy.clone()  # residual
    p = r.clone()
    rsold = torch.sum(torch.conj(r) * r).real

    for i in range(num_iters):
        Ap = At(A(p))
        alpha = rsold / torch.sum(torch.conj(p) * Ap).real
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(torch.conj(r) * r).real
        if torch.sqrt(rsnew) < 1e-6:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        print(f"Iter {i+1}, Residual: {torch.sqrt(rsnew):.6e}")

    return x

# -------- Example usage --------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate radial trajectory
    num_spokes = 64
    samples_per_spoke = 256
    Nx, Ny = 128, 128

    angles = torch.linspace(0, np.pi, num_spokes, device=device)
    radii = torch.linspace(-0.5, 0.5, samples_per_spoke, device=device)
#    kx = torch.cat([r * torch.cos(theta).repeat(samples_per_spoke) for theta in angles])
#    ky = torch.cat([r * torch.sin(theta).repeat(samples_per_spoke) for theta in angles])
    kx = torch.cat([radii * torch.cos(theta) for theta in angles])
    ky = torch.cat([radii * torch.sin(theta) for theta in angles])


    # Simulated k-space data: Gaussian blob
    kspace_data = torch.exp(-100 * (kx**2 + ky**2)).to(device).to(torch.complex64)

    # Run iterative reconstruction
    recon_img = iterative_recon(kx, ky, kspace_data, (Nx, Ny), oversamp=2.0, num_iters=10)

    import matplotlib.pyplot as plt
    plt.imshow(recon_img.abs().cpu().numpy(), cmap='gray')
    plt.title("Iterative NUFFT Reconstruction")
    plt.axis('off')
    plt.show()












def simple_phantom(size=128, device='cpu'):
    Y, X = torch.meshgrid(torch.linspace(-1, 1, size, device=device),
                          torch.linspace(-1, 1, size, device=device), indexing='ij')
    phantom = torch.zeros_like(X)

    # Add ellipses like Shepp-Logan
    phantom += 1.0 * (((X)**2 + (Y/1.5)**2) <= 0.9**2).float()
    phantom -= 0.8 * (((X+0.3)**2 + (Y/1.5)**2) <= 0.4**2).float()
    phantom += 0.5 * (((X-0.2)**2 + (Y-0.2)**2) <= 0.2**2).float()

    return phantom





import torch
import numpy as np
import matplotlib.pyplot as plt
#from skimage.data import shepp_logan_phantom
#from skimage.transform import resize

# 1. Generate Shepp-Logan phantom
def generate_phantom(size=128, device='cpu'):
    phantom_resized = simple_phantom(); #hepp_logan_phantom()
    #phantom_resized = resize(phantom, (size, size), mode='reflect', anti_aliasing=True)
    return torch.tensor(phantom_resized, dtype=torch.float32, device=device)

# Load or create your NUFFT functions here (from previous steps):
# - nufft2d2_forward
# - nufft2d2_adjoint
# - iterative_recon
# - kaiser_bessel_kernel
# - estimate_density_compensation

# 2. Generate k-space trajectory (radial)
def generate_radial_trajectory(num_spokes=64, samples_per_spoke=256, device='cpu'):
    angles = torch.linspace(0, np.pi, num_spokes, device=device)
    radii = torch.linspace(-0.5, 0.5, samples_per_spoke, device=device)
    radii_grid, angles_grid = torch.meshgrid(radii, angles, indexing='ij')
    kx = (radii_grid * torch.cos(angles_grid)).reshape(-1)
    ky = (radii_grid * torch.sin(angles_grid)).reshape(-1)
    return kx, ky

# --- MAIN EXECUTION ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Nx = Ny = 128

# Step 1: Phantom
phantom = generate_phantom(Nx, device=device)

# Step 2: k-space trajectory
num_spokes = 128
samples_per_spoke = 256
kx, ky = generate_radial_trajectory(num_spokes, samples_per_spoke, device=device)

# Step 3: Simulate k-space data via forward NUFFT
phantom_complex = phantom.to(torch.complex64)
kspace_data = nufft2d2_forward(kx, ky, phantom_complex, oversamp=2.0)

# Step 4: Reconstruct image from non-uniform k-space
recon = iterative_recon(kx, ky, kspace_data, (Nx, Ny), oversamp=2.0, num_iters=10)

# Step 5: Display
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(phantom.cpu(), cmap='gray')
axs[0].set_title("Original Shepp-Logan Phantom")
axs[0].axis('off')

axs[1].imshow(recon.abs().cpu(), cmap='gray')
axs[1].set_title("Reconstructed Image")
axs[1].axis('off')

plt.show()







