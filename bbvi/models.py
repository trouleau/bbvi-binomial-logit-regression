import torch

from .posteriors import Posterior
from .priors import Prior


class ModelBinomialLogit:

    def __init__(self):
        pass

    def set_data(self, covariate_df, response_df):
        self.columns = list(covariate_df.columns)
        self.X = torch.tensor(covariate_df.to_numpy(), dtype=torch.double)
        self.n_points = self.X.shape[0]
        self.n_params = self.X.shape[1]
        self.y = torch.tensor(response_df['positive_count'].to_numpy(),
                              dtype=torch.double)
        if len(self.y) != self.n_points:
            raise ValueError("Mismatch input shapes")
        self.n = torch.tensor(response_df['total_count'].to_numpy(),
                              dtype=torch.double)
        if len(self.n) != self.n_points:
            raise ValueError("Mismatch input shapes")
        self._init_cache()

    def _init_cache(self):
        log_factorial_n = torch.lgamma(self.n + 1)
        log_factorial_k = torch.lgamma(self.y + 1)
        log_factorial_nmk = torch.lgamma(self.n - self.y + 1)
        self._ll_const = log_factorial_n - log_factorial_k - log_factorial_nmk

    def log_likelihood(self, coeffs):
        # Compute the logits from the coefficients `coeffs`
        eta = self.X.mm(coeffs.unsqueeze(-1)).squeeze()
        # Compute the log-likelihood at each data point
        val = self._ll_const + self.y * eta - self.n * torch.log1p(eta.exp())
        return val.sum()

    def objective(self, beta):
        return -1.0 * self.log_likelihood(beta)


class ModelVariationalBinomialLogit(ModelBinomialLogit):

    def __init__(self, posterior, prior, n_samples):
        # Validate and set the posterior attribute
        if not isinstance(posterior, Posterior):
            raise ValueError("`posterior` should be a `Posterior` object")
        self.posterior = posterior
        # Validate and set the prior attribute
        if not isinstance(prior, Prior):
            raise ValueError("`prior` should be a `Prior` object")
        self.prior = prior
        # Validate and set the device
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # Set the number of samples
        self.n_samples = n_samples
        super().__init__()

    def objective(self, gamma):
        # Split the parameters into `loc` and `log_scale`
        loc, log_scale = gamma[:self.n_params], gamma[self.n_params:]
        # Sample noise
        sample_size = (self.n_samples, self.n_params)
        eps_arr = self.posterior.sample_epsilon(size=sample_size)
        # Reparametrize noise with posterior
        z_arr = self.posterior.g(eps_arr, loc, log_scale)
        # Initialize the output variable
        ll_value = 0.0
        for l in range(self.n_samples):
            ll_value += self.log_likelihood(z_arr[l])
        ll_value /= self.n_samples
        prior_value = 0.0
        for l in range(self.n_samples):
            prior_value += self.prior.logprob(z_arr[l])
        prior_value /= self.n_samples
        entropy_value = self.posterior.entropy(loc, log_scale)
        value = -1.0 * (ll_value + prior_value - entropy_value)
        self.debug = {
            'll_value': ll_value,
            'prior_value': prior_value,
            'entropy_value': entropy_value
        }
        return value
