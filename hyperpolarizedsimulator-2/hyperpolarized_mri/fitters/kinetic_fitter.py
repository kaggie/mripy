import torch
from torch.optim import Adam

class KineticFitter:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def fit(self, time, data, params_init, n_iter=200):
        """
        Fit model parameters to data using PyTorch autograd.
        """
        params = {k: torch.tensor(float(v), requires_grad=True) for k, v in params_init.items()}
        optimizer = Adam(params.values(), lr=self.lr)

        for _ in range(n_iter):
            optimizer.zero_grad()
            sim = self.model(time, params)
            loss = torch.mean((sim - data)**2)
            loss.backward()
            optimizer.step()
        fitted = {k: v.detach().item() for k, v in params.items()}
        return fitted