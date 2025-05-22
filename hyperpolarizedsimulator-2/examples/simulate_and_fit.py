import torch
import numpy as np
from hyperpolarized_mri.models.two_site import TwoSiteExchangeModel
from hyperpolarized_mri.simulators.forward_simulator import ForwardSimulator
from hyperpolarized_mri.fitters.kinetic_fitter import KineticFitter
from hyperpolarized_mri.visualization.plot_signals import plot_signals

# Define time points
time = torch.linspace(0, 60, 61)

# True parameters
params_true = {
    'kAB': 0.05, 'kBA': 0.02,
    'T1A': 30, 'T1B': 25,
    'M0A': 1.0, 'M0B': 0.0
}

# Simulate data
model = TwoSiteExchangeModel()
simulator = ForwardSimulator(model)
signals = simulator.simulate(time, params_true).detach().numpy()
plot_signals(time, signals, labels=['A', 'B'], title='Ground Truth Signals')

# Add noise
np.random.seed(0)
signals_noisy = signals + 0.03*np.random.randn(*signals.shape)

# Fit
params_init = {k: v*0.8 for k, v in params_true.items()}
fitter = KineticFitter(model)
fitted_params = fitter.fit(time, torch.tensor(signals_noisy), params_init)
print("Fitted Params:", fitted_params)

# Simulate fitted
signals_fit = simulator.simulate(time, fitted_params).detach().numpy()
plot_signals(time, signals_fit, labels=['A', 'B'], title='Fitted Signals')