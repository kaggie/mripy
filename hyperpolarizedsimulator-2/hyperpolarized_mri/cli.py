import argparse
import torch
from hyperpolarized_mri.models.two_site import TwoSiteExchangeModel
from hyperpolarized_mri.simulators.forward_simulator import ForwardSimulator

def main():
    parser = argparse.ArgumentParser(description='Hyperpolarized MRI Toolbox CLI')
    parser.add_argument('--model', type=str, default='two_site', help='Model: two_site/three_site')
    parser.add_argument('--tmax', type=float, default=60)
    parser.add_argument('--dt', type=float, default=1)
    args = parser.parse_args()
    time = torch.arange(0, args.tmax+args.dt, args.dt)
    params = {
        'kAB': 0.05, 'kBA': 0.02,
        'T1A': 30, 'T1B': 25,
        'M0A': 1.0, 'M0B': 0.0
    }
    if args.model == 'two_site':
        model = TwoSiteExchangeModel()
    # Add more models here as needed
    simulator = ForwardSimulator(model)
    signals = simulator.simulate(time, params).detach().numpy()
    print('Simulated signal:', signals)

if __name__ == '__main__':
    main()