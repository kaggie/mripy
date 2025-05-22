import torch
import torch.optim as optim
import torch.nn as nn # For MSELoss
import numpy as np

# Attempt relative imports for plugin structure
try:
    from .model import MRSIModel
    from .basis import BasisSet, load_basis_spectrum_csv # For testing block
except ImportError:
    # Fallback for direct script execution (e.g., during development/testing)
    # This assumes 'model.py' and 'basis.py' are in the same directory as 'fitter.py'
    # or the parent directory is in PYTHONPATH.
    print("Attempting fallback imports for MRSIModel and BasisSet.")
    from model import MRSIModel
    from basis import BasisSet, load_basis_spectrum_csv


def fit_spectrum(measured_spectrum_tensor, basis_set, num_baseline_coeffs, 
                 num_iterations=1000, learning_rate=0.01, print_loss_every=100):
    """
    Fits a measured spectrum using the MRSIModel and a given basis set.

    Args:
        measured_spectrum_tensor (torch.Tensor): A 1D PyTorch tensor representing the
                                                 spectrum to be fitted. Assumed to be complex.
        basis_set (BasisSet): An instance of the BasisSet class containing the
                              metabolite basis spectra.
        num_baseline_coeffs (int): Number of coefficients for the polynomial baseline.
        num_iterations (int, optional): Number of iterations for optimization.
                                        Defaults to 1000.
        learning_rate (float, optional): Learning rate for the Adam optimizer.
                                         Defaults to 0.01.
        print_loss_every (int, optional): Print loss every N iterations. If 0 or None,
                                          no loss is printed during training. Defaults to 100.

    Returns:
        tuple: A tuple containing:
            - model (MRSIModel): The trained model with fitted parameters.
            - final_loss (float): The loss value at the end of training.
    
    Raises:
        TypeError: If inputs are not of the expected type.
        ValueError: If input dimensions or values are inconsistent.
    """
    if not isinstance(measured_spectrum_tensor, torch.Tensor):
        raise TypeError("measured_spectrum_tensor must be a PyTorch Tensor.")
    if not torch.is_complex(measured_spectrum_tensor):
        # For now, strictly complex. Could be adapted.
        raise ValueError("measured_spectrum_tensor must be complex-valued for this version of fit_spectrum.")
    if measured_spectrum_tensor.ndim != 1:
        raise ValueError("measured_spectrum_tensor must be a 1D tensor.")

    if not isinstance(basis_set, BasisSet):
        raise TypeError("basis_set must be an instance of the BasisSet class.")
    if basis_set.num_metabolites() == 0 or basis_set.num_points() == 0:
        raise ValueError("BasisSet cannot be empty (no metabolites or zero points).")
    
    if measured_spectrum_tensor.shape[0] != basis_set.num_points():
        raise ValueError(f"Measured spectrum length ({measured_spectrum_tensor.shape[0]}) "
                         f"must match basis set spectral points ({basis_set.num_points()}).")

    # Convert basis spectra NumPy array to PyTorch tensor
    try:
        basis_spectra_np = basis_set.get_spectra_array()
        basis_spectra_tensor = torch.from_numpy(basis_spectra_np).float()
        # Ensure it's on the same device as the measured spectrum
        basis_spectra_tensor = basis_spectra_tensor.to(measured_spectrum_tensor.device)
    except Exception as e:
        raise ValueError(f"Could not convert basis_set to tensor: {e}")

    # Initialize Model
    model = MRSIModel(
        num_basis_spectra=basis_set.num_metabolites(),
        num_spectral_points=basis_set.num_points(),
        num_baseline_coeffs=num_baseline_coeffs
    )
    # Move model to the same device as the measured spectrum
    model.to(measured_spectrum_tensor.device)
    model.train() # Set model to training mode

    # Define Loss Function
    # MSELoss for complex numbers: mean(abs(target - input)**2)
    loss_fn = nn.MSELoss()

    # Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Optimization Loop
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed_spectrum = model(basis_spectra_tensor)
        
        # Calculate loss
        loss = loss_fn(reconstructed_spectrum, measured_spectrum_tensor)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if print_loss_every and (iteration + 1) % print_loss_every == 0:
            print(f"Iteration [{iteration+1}/{num_iterations}], Loss: {loss.item():.6f}")

    final_loss = loss.item()
    print(f"Optimization finished. Final Loss: {final_loss:.6f}")
    
    model.eval() # Set model to evaluation mode
    return model, final_loss


if __name__ == '__main__':
    import os
    import csv

    # --- Create Dummy BasisSet for testing ---
    dummy_basis_dir = "temp_dummy_basis_for_fitter_test"
    if not os.path.exists(dummy_basis_dir):
        os.makedirs(dummy_basis_dir)

    num_points_test = 512
    num_metabolites_test = 3
    basis_names = []
    basis_spectra_list_np = []

    # Create dummy basis CSV files
    for i in range(num_metabolites_test):
        name = f"Metabolite_{i+1}"
        basis_names.append(name)
        # Simple Gaussian-like peak for each basis spectrum at different positions
        center = (i + 1) * num_points_test / (num_metabolites_test + 1)
        x = np.arange(num_points_test)
        spectrum_np = np.exp(-((x - center)**2) / (2 * (num_points_test/20)**2))
        basis_spectra_list_np.append(spectrum_np)
        
        filepath = os.path.join(dummy_basis_dir, f"{name}.csv")
        with open(filepath, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frequency", "Intensity"]) # Header
            for val in spectrum_np: # Only write intensity for simplicity
                writer.writerow([0.0, f"{val:.6f}"]) # Dummy frequency

    # Create BasisSet instance (using the direct constructor for simplicity here)
    test_basis_set = BasisSet(names=basis_names, spectra=[np.array(s) for s in basis_spectra_list_np])
    print(f"Created dummy BasisSet: {test_basis_set}")


    # --- Create a Dummy Measured Spectrum (complex) ---
    # This could be a linear combination of basis spectra + baseline + noise + phase error
    true_concentrations = torch.tensor([0.8, 1.2, 0.5], dtype=torch.float32)
    true_phase_rad = torch.tensor(np.deg2rad(15.0), dtype=torch.float32) # 15 degrees phase error
    
    basis_tensor_np = test_basis_set.get_spectra_array()
    basis_tensor_torch = torch.from_numpy(basis_tensor_np).float()

    # True metabolite sum (real)
    true_met_sum = torch.matmul(true_concentrations, basis_tensor_torch)

    # True baseline (real)
    true_num_baseline_coeffs = 3 # quadratic
    x_poly_true = torch.linspace(-1, 1, num_points_test, dtype=torch.float32)
    poly_terms_true = torch.stack([x_poly_true**i for i in range(true_num_baseline_coeffs)], dim=0)
    true_baseline_coeffs_vals = torch.tensor([0.1, -0.05, 0.02], dtype=torch.float32) # Example coeffs
    true_baseline = torch.matmul(true_baseline_coeffs_vals, poly_terms_true)
    
    true_real_part = true_met_sum + true_baseline
    
    # Apply phase to make it complex
    exp_phase_true = torch.complex(torch.cos(true_phase_rad), torch.sin(true_phase_rad))
    measured_spectrum_complex = true_real_part * exp_phase_true
    
    # Add some noise
    noise_real = torch.randn(num_points_test) * 0.02
    noise_imag = torch.randn(num_points_test) * 0.02
    measured_spectrum_complex += torch.complex(noise_real, noise_imag)
    
    print(f"Created dummy measured_spectrum_tensor (complex) with shape: {measured_spectrum_complex.shape}")

    # --- Test fit_spectrum function ---
    print("\\n--- Testing fit_spectrum ---")
    num_fit_baseline_coeffs = 3 # Match true for easier check, but can be different
    fit_iterations = 2000
    fit_lr = 0.01

    try:
        fitted_model, final_loss = fit_spectrum(
            measured_spectrum_tensor=measured_spectrum_complex,
            basis_set=test_basis_set,
            num_baseline_coeffs=num_fit_baseline_coeffs,
            num_iterations=fit_iterations,
            learning_rate=fit_lr,
            print_loss_every=500
        )
        
        print("\\n--- Fit Results ---")
        print(f"Final loss: {final_loss:.6f}")
        print("Fitted Parameters:")
        for name, param in fitted_model.named_parameters():
            if name == 'concentrations':
                print(f"  {name}: {param.data.tolist()}")
                print(f"  (True concentrations for comparison: {true_concentrations.tolist()})")
            elif name == 'baseline_coeffs':
                print(f"  {name}: {param.data.tolist()}")
                if num_fit_baseline_coeffs == true_num_baseline_coeffs:
                     print(f"  (True baseline_coeffs for comparison: {true_baseline_coeffs_vals.tolist()})")
            elif name == 'phase_rad':
                print(f"  {name} (rad): {param.data.item():.4f}")
                print(f"  {name} (deg): {np.rad2deg(param.data.item()):.2f}")
                print(f"  (True phase_rad for comparison (deg): {np.rad2deg(true_phase_rad.item()):.2f})")
        
        # Simple assertions
        assert final_loss < 0.01, f"Final loss {final_loss} is too high. Fit might be poor."
        # Check if concentrations are somewhat close (very loose check)
        fitted_concs = fitted_model.concentrations.data
        diff_concs = torch.abs(fitted_concs - true_concentrations).mean()
        assert diff_concs < 0.5, f"Concentration difference {diff_concs} is too high."

    except Exception as e:
        print(f"Error during fit_spectrum test: {e}")
        raise # Re-raise to see traceback if something unexpected happens

    finally:
        # Clean up dummy directory
        import shutil
        if os.path.exists(dummy_basis_dir):
            shutil.rmtree(dummy_basis_dir)
            print(f"Cleaned up dummy directory: {dummy_basis_dir}")

    print("\\nfit_spectrum test finished.")
```
