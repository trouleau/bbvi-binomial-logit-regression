import math
import torch


class Posterior:

    def __init__(self, device='cpu'):
        self.device = device

    def sample_epsilon(self, size):
        """
        Sample an array of epsilons with shape `size`
        """
        raise NotImplementedError('Must be implemented in child class')

    def g(self, eps, loc, log_scale):
        """
        Reparamaterization of the approximate posterior function where `loc` is
        the location of the distribution (i.e. usually its mean) and
        `log_scale` is the log of its scale (i.e. usually its standard
        deviation). Note: the log is optimized for better stability in the
        optimization
        """
        raise NotImplementedError('Must be implemented in child class')

    def logpdf(self, eps, loc, log_scale):
        """
        Log-density function parameterized by `loc` and `log_scale` and
        evaluated at `eps`
        """
        raise NotImplementedError('Must be implemented in child class')

    def entropy(self, loc, log_scale):
        """
        Entropy of the distribution parameterized by `loc` and `log_scale`
        """
        raise NotImplementedError('Must be implemented in child class')


class GaussianPosterior(Posterior):

    _const_logpdf = -0.5 * math.log(2 * math.pi)
    _const_entropy = 0.5 * math.log(2 * math.pi * math.exp(1))

    def sample_epsilon(self, size):
        """
        Sample an array of epsilons from normal distribution with shape `size`
        """
        return torch.randn(size, dtype=torch.float64, device=self.device,
                           requires_grad=False)

    def g(self, eps, loc, log_scale):
        """
        Reparamaterization of the approximate Gaussian posterior function
        """
        return loc + eps * log_scale.exp()

    def logpdf(self, eps, loc, log_scale):
        """
        Log-density function parameterized by `loc` and `log_scale` and
        evaluated at `eps`
        """
        z = self.g(eps, loc, log_scale)
        var = log_scale.exp() ** 2
        return self._const_logpdf - log_scale - ((z - loc) ** 2) / (2 * var)

    def entropy(self, loc, log_scale):
        return (self._const_entropy + log_scale).sum()
