import torch


class Prior:

    def __init__(self, C):
        if not isinstance(C, torch.Tensor):
            raise ValueError("Parameter `C` should be a tensor")
        self.C = C

    def logprob(self, z):
        """
        Log Prior
        """
        raise NotImplementedError("Must be implemented in child class")


class GaussianPrior(Prior):
    """
    Gaussian Prior centered in zero, with std `C`.

    """

    def logprob(self, z):
        """
        Log Prior
        """
        return -torch.sum((z ** 2) / self.C)
