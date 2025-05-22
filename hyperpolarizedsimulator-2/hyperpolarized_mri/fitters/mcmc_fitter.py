import torch
import numpy as np

class SimpleMCMCFitter:
    """
    Very basic Metropolis-Hastings MCMC for parameter estimation.
    """
    def __init__(self, model, n_samples=1000, noise_std=0.05):
        self.model = model
        self.n_samples = n_samples
        self.noise_std = noise_std

    def fit(self, time, data, params_init, step_size=0.1):
        params = {k: float(v) for k, v in params_init.items()}
        samples = []
        current_params = params.copy()
        current_sim = self.model(time, current_params).detach().numpy()
        current_loss = np.mean((data - current_sim)**2)
        for _ in range(self.n_samples):
            proposal = {k: v + np.random.normal(scale=step_size) for k, v in current_params.items()}
            sim = self.model(time, proposal).detach().numpy()
            loss = np.mean((data - sim)**2)
            accept_prob = min(1, np.exp((current_loss - loss) / (2 * self.noise_std**2)))
            if np.random.rand() < accept_prob:
                current_params = proposal
                current_loss = loss
            samples.append(current_params.copy())
        return samples