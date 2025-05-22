import torch
import torch.nn as nn
# Changed from relative to direct import
from operators import LinearOperator # Used for type hinting and base class reference

class L2Reconstruction(nn.Module):
    def __init__(self, linear_operator: LinearOperator, num_iterations=10, learning_rate=0.1):
        super().__init__()
        self.linear_operator = linear_operator
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

    def forward(self, y_kspace):
        # Initialize x_recon with appropriate shape and dtype
        # We can get the image shape by performing an adjoint operation on y_kspace once
        # This is a common way to estimate the image domain size if not explicitly known
        with torch.no_grad(): # Ensure this operation does not track gradients
            x_initial_guess = self.linear_operator.adjoint(y_kspace)
        
        x_recon = torch.zeros_like(x_initial_guess, requires_grad=True)

        for _ in range(self.num_iterations):
            # Calculate residual: residual = Ax - y
            residual = self.linear_operator.forward(x_recon) - y_kspace
            
            # Calculate gradient: grad = A^H(Ax - y)
            # The gradient calculation itself should not be part of the graph we differentiate for x_recon's update
            with torch.no_grad():
                grad = self.linear_operator.adjoint(residual)

            # Update x_recon
            # x_recon = x_recon - self.learning_rate * grad
            # To avoid issues with in-place modification of a leaf variable,
            # we can use torch.no_grad() or manage detach() and requires_grad_(True)

            # Debugging for NUFFT NaN (commented out for cleaner output)
            # if isinstance(self.linear_operator, torch.nn.Module) and "NUFFT" in self.linear_operator.__class__.__name__: # Only print for NUFFT
            #     if _ == 0: print(f"L2Recon NUFFT Debug: Initial x_recon norm: {torch.norm(x_recon)}")
            #     if torch.isnan(grad).any() or torch.isinf(grad).any():
            #         print(f"Iteration {_}: grad has nan/inf. Norm: {torch.norm(grad)}")
            #         print(f"Iteration {_}: residual norm: {torch.norm(residual)}")
            #     if torch.isnan(x_recon).any() or torch.isinf(x_recon).any():
            #         print(f"Iteration {_}: x_recon has nan/inf BEFORE update. Norm: {torch.norm(x_recon)}")

            # Option 1: Using torch.no_grad() context
            with torch.no_grad():
                 x_recon -= self.learning_rate * grad
            
            # if isinstance(self.linear_operator, torch.nn.Module) and "NUFFT" in self.linear_operator.__class__.__name__: # Only print for NUFFT
            #     if torch.isnan(x_recon).any() or torch.isinf(x_recon).any():
            #         print(f"Iteration {_}: x_recon has nan/inf AFTER update. Norm: {torch.norm(x_recon)}")
            #         # break # Stop if NaN occurs to inspect last valid state
            #     if _ % 10 == 0 : print(f"Iter {_}: x_recon norm: {torch.norm(x_recon)}, grad norm: {torch.norm(grad)}")

            # After update, ensure requires_grad is True for the next iteration's forward pass
            x_recon.requires_grad_(True)

            # Option 2: (Alternative, more explicit)
            # x_recon_prev = x_recon.detach().clone() # Create a detached copy
            # x_recon = x_recon_prev - self.learning_rate * grad
            # x_recon.requires_grad_(True) # Enable gradient tracking for the new tensor

        return x_recon.detach()

class L1Reconstruction(nn.Module):
    def __init__(self, linear_operator: LinearOperator, num_iterations=10, learning_rate=0.1, lambda_reg=0.01):
        super().__init__()
        self.linear_operator = linear_operator
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate # Step size alpha for ISTA
        self.lambda_reg = lambda_reg

    def _soft_threshold(self, x, threshold):
        abs_x = torch.abs(x)
        # max_val = torch.maximum(abs_x - threshold, torch.zeros_like(abs_x))
        # A simpler way to write max(val, 0) for real tensors is torch.relu
        max_val = torch.relu(abs_x - threshold) 

        if x.is_complex():
            # torch.sgn(x) handles x=0 correctly (returns 0+0j)
            return torch.sgn(x) * max_val
        else:
            return torch.sign(x) * max_val

    def forward(self, y_kspace):
        # Initialize x_recon
        with torch.no_grad(): # Adjoint for shape inference should not track gradients
            x_initial_guess = self.linear_operator.adjoint(y_kspace)
        x_recon = torch.zeros_like(x_initial_guess, requires_grad=False) # ISTA updates are explicit

        for _ in range(self.num_iterations):
            # Gradient step: x_grad_update = x_recon - alpha * A^H(A(x_recon) - y_kspace)
            # All operations here are part of the ISTA algorithm, not for autograd of x_recon itself.
            with torch.no_grad():
                residual = self.linear_operator.forward(x_recon) - y_kspace
                grad = self.linear_operator.adjoint(residual)
                x_grad_update = x_recon - self.learning_rate * grad
            
            # Proximal step (soft thresholding)
            # x_recon = soft_thresh(x_grad_update, alpha * lambda)
            threshold_val = self.learning_rate * self.lambda_reg
            x_recon = self._soft_threshold(x_grad_update, threshold_val)
            
        return x_recon
