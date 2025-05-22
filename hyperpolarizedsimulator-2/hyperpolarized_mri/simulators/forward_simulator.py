import torch

class ForwardSimulator:
    """
    Class to simulate signal timecourses from a kinetic model.
    """
    def __init__(self, model):
        self.model = model

    def simulate(self, time, params):
        """
        Simulate signal using provided model and parameters.
        """
        return self.model(time, params)