import torch
import torch.nn as nn

class MRSIModel(nn.Module):
    """
    A PyTorch model for Magnetic Resonance Spectroscopy Imaging (MRSI) data fitting.

    This model reconstructs a spectrum by combining basis spectra of metabolites
    with learned concentrations, adding a polynomial baseline, and applying a
    zero-order phase correction.

    Assumptions:
        - Basis spectra provided to the forward pass are real-valued (e.g., magnitude
          spectra or the real part after prior phasing).
        - The model initially implements a simple zero-order phase correction.
    """
    def __init__(self, num_basis_spectra, num_spectral_points, num_baseline_coeffs):
        """
        Initializes the MRSIModel.

        Args:
            num_basis_spectra (int): The number of metabolite basis spectra.
                                     This determines the size of the 'concentrations' parameter.
            num_spectral_points (int): The number of points in each spectrum.
            num_baseline_coeffs (int): The number of coefficients for the polynomial
                                       baseline (e.g., polynomial_order + 1).
        """
        super().__init__()

        if not isinstance(num_basis_spectra, int) or num_basis_spectra <= 0:
            raise ValueError("num_basis_spectra must be a positive integer.")
        if not isinstance(num_spectral_points, int) or num_spectral_points <= 0:
            raise ValueError("num_spectral_points must be a positive integer.")
        if not isinstance(num_baseline_coeffs, int) or num_baseline_coeffs < 0:
            # Allow 0 if no baseline is desired, though typically at least 1 (offset)
            raise ValueError("num_baseline_coeffs must be a non-negative integer.")

        self.num_spectral_points = num_spectral_points
        self.num_basis_spectra = num_basis_spectra
        self.num_baseline_coeffs = num_baseline_coeffs

        # Learnable parameters
        # Concentrations (c_m): Initialize with small positive random values or ones.
        # Using ones for simplicity and to ensure initial contribution.
        self.concentrations = nn.Parameter(torch.ones(num_basis_spectra))
        
        # Baseline Coefficients: Initialize with zeros or small random values.
        # Zeros make the initial baseline flat at zero.
        if num_baseline_coeffs > 0:
            self.baseline_coeffs = nn.Parameter(torch.zeros(num_baseline_coeffs))
        else:
            # If no baseline coeffs, create an empty parameter or handle in forward pass
            # For simplicity, we'll assume num_baseline_coeffs > 0 if baseline is used.
            # Or, register as None and check in forward. For now, rely on num_baseline_coeffs > 0.
            self.baseline_coeffs = None


        # Phase (Zero-order phase, scalar): Initialize to 0.0.
        self.phase_rad = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Create polynomial basis for the baseline
        # This is fixed and does not require gradients.
        if num_baseline_coeffs > 0:
            # x_poly ranges from -1 to 1, providing a normalized domain for polynomial fitting
            x_poly = torch.linspace(-1, 1, num_spectral_points, dtype=torch.float32)
            
            # Create polynomial terms: [x^0, x^1, x^2, ..., x^(N-1)]
            # Each term is a row: shape (num_baseline_coeffs, num_spectral_points)
            polynomial_terms_list = [x_poly**i for i in range(num_baseline_coeffs)]
            polynomial_terms_tensor = torch.stack(polynomial_terms_list, dim=0)
            self.register_buffer('polynomial_terms', polynomial_terms_tensor)
        else:
            self.register_buffer('polynomial_terms', torch.empty(0, num_spectral_points, dtype=torch.float32))


    def forward(self, basis_spectra_tensor):
        """
        Performs the forward pass of the model.

        Args:
            basis_spectra_tensor (torch.Tensor): A 2D tensor of the basis spectra,
                                                 with shape (num_basis_spectra, num_spectral_points).
                                                 These are assumed to be real-valued.

        Returns:
            torch.Tensor: The reconstructed complex spectrum, with shape (num_spectral_points,).
        """
        if not isinstance(basis_spectra_tensor, torch.Tensor):
            raise TypeError("basis_spectra_tensor must be a PyTorch Tensor.")
        if basis_spectra_tensor.ndim != 2:
            raise ValueError(f"basis_spectra_tensor must be 2D, but got shape {basis_spectra_tensor.shape}")
        if basis_spectra_tensor.shape[0] != self.num_basis_spectra:
            raise ValueError(f"basis_spectra_tensor first dimension ({basis_spectra_tensor.shape[0]}) "
                             f"does not match num_basis_spectra ({self.num_basis_spectra}).")
        if basis_spectra_tensor.shape[1] != self.num_spectral_points:
            raise ValueError(f"basis_spectra_tensor second dimension ({basis_spectra_tensor.shape[1]}) "
                             f"does not match num_spectral_points ({self.num_spectral_points}).")

        # Ensure concentrations are positive (optional, can be enforced by optimizer or loss)
        # concentrations_positive = torch.relu(self.concentrations) + 1e-6 # Small epsilon to avoid zero
        concentrations_to_use = self.concentrations # Using raw concentrations for now

        # Calculate Metabolite Sum
        # concentrations: (num_basis_spectra)
        # basis_spectra_tensor: (num_basis_spectra, num_spectral_points)
        # metabolite_sum: (num_spectral_points)
        metabolite_sum = torch.matmul(concentrations_to_use, basis_spectra_tensor)

        # Calculate Baseline
        if self.num_baseline_coeffs > 0 and self.baseline_coeffs is not None:
            # baseline_coeffs: (num_baseline_coeffs)
            # polynomial_terms: (num_baseline_coeffs, num_spectral_points)
            # estimated_baseline: (num_spectral_points)
            estimated_baseline = torch.matmul(self.baseline_coeffs, self.polynomial_terms)
            reconstructed_spectrum_real = metabolite_sum + estimated_baseline
        else:
            reconstructed_spectrum_real = metabolite_sum # No baseline calculation

        # Apply Phase Correction (Simple Zero-Order)
        # reconstructed_spectrum_real: (num_spectral_points)
        # phase_rad: scalar
        # exp_phase: scalar complex
        # reconstructed_spectrum_complex: (num_spectral_points) complex
        
        # Create complex exponential for phasing
        # Ensure phase_rad is on the same device as reconstructed_spectrum_real
        # Using torch.complex for explicit complex number creation
        exp_phase = torch.complex(torch.cos(self.phase_rad), torch.sin(self.phase_rad))
        
        # Broadcasting real spectrum with complex phase factor
        reconstructed_spectrum_complex = reconstructed_spectrum_real * exp_phase
        
        return reconstructed_spectrum_complex

if __name__ == '__main__':
    # --- Example Usage and Testing ---
    num_metabolites = 5
    num_points = 1024
    num_coeffs_baseline = 3 # Quadratic baseline (order 2 -> 3 coeffs: c0, c1, c2)

    print(f"Creating MRSIModel with {num_metabolites} metabolites, "
          f"{num_points} spectral points, and {num_coeffs_baseline} baseline coefficients.\\n")

    # Instantiate the model
    model = MRSIModel(num_basis_spectra=num_metabolites,
                      num_spectral_points=num_points,
                      num_baseline_coeffs=num_coeffs_baseline)
    
    print("Model instantiated successfully.")
    print("Model parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  Name: {name}, Size: {param.size()}, Initial Value (sample): {param.data.flatten()[0].item():.4f}")
    
    print("\\nRegistered buffers:")
    for name, buf in model.named_buffers():
        print(f"  Name: {name}, Size: {buf.size()}")
        if name == 'polynomial_terms':
            print(f"    Polynomial terms sample (first term, first 5 points): {buf[0, :5].tolist()}")
            if buf.shape[0] > 1:
                 print(f"    Polynomial terms sample (second term, first 5 points): {buf[1, :5].tolist()}")


    # Create a dummy basis spectra tensor (real-valued)
    # Shape: (num_metabolites, num_points)
    dummy_basis_spectra = torch.rand(num_metabolites, num_points, dtype=torch.float32)
    print(f"\\nCreated dummy basis spectra tensor with shape: {dummy_basis_spectra.shape}")

    # Perform a forward pass
    print("\\nPerforming a forward pass...")
    try:
        reconstructed_output = model(dummy_basis_spectra)
        print(f"Forward pass successful. Output shape: {reconstructed_output.shape}, "
              f"Output dtype: {reconstructed_output.dtype}")
        assert reconstructed_output.shape == (num_points,), "Output shape mismatch"
        assert reconstructed_output.is_complex(), "Output should be complex"
        print(f"Output sample (first 5 points, real part): {reconstructed_output.real[:5].tolist()}")
        print(f"Output sample (first 5 points, imag part): {reconstructed_output.imag[:5].tolist()}")

    except Exception as e:
        print(f"Error during forward pass: {e}")

    # Test with zero baseline coefficients
    print("\\n--- Test with zero baseline coefficients ---")
    model_no_baseline = MRSIModel(num_basis_spectra=num_metabolites,
                                  num_spectral_points=num_points,
                                  num_baseline_coeffs=0)
    print("Model (no baseline) instantiated.")
    try:
        reconstructed_no_baseline = model_no_baseline(dummy_basis_spectra)
        print(f"Forward pass (no baseline) successful. Output shape: {reconstructed_no_baseline.shape}")
        assert reconstructed_no_baseline.shape == (num_points,), "No baseline output shape mismatch"
        
        # Check if baseline_coeffs is None or handled
        assert model_no_baseline.baseline_coeffs is None, "baseline_coeffs should be None for num_coeffs=0"
        assert model_no_baseline.polynomial_terms.numel() == 0, "polynomial_terms should be empty for num_coeffs=0"

    except Exception as e:
        print(f"Error during forward pass (no baseline): {e}")

    print("\\nAll tests finished.")
```
