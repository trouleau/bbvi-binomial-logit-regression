import numpy as np
import scipy.stats
import torch

import bbvi.posteriors


if __name__ == "__main__":

    # Define sample size
    N = 20
    L = 100

    # Define distribution parameters
    mu = torch.tensor(np.random.uniform(-10.0, 10.0, size=N),
                      dtype=torch.float64).unsqueeze(-1)
    std = torch.tensor(np.random.uniform(0.0, 10.0, size=N),
                       dtype=torch.float64).unsqueeze(-1)
    log_std = std.log()

    gaussian_post_obj = posteriors.GaussianPosterior()

    # Test the log-pdf function
    # =========================
    print("Test `sample_epsilon` and `logpdf`...")
    # Sample from the posterior object
    eps_arr = gaussian_post_obj.sample_epsilon(size=(N, L))
    y_mine = gaussian_post_obj.logpdf(eps_arr, loc=mu, log_scale=log_std)
    # Compare with the scipy implementation
    y_scipy = scipy.stats.norm.logpdf(mu + eps_arr * std, loc=mu, scale=std)
    assert np.allclose(y_mine, y_scipy)
    print("ok!")

    # Test the entropy function
    # =========================
    print("Test `entropy`...")
    # Compute using the posterior object
    entropy_mine = gaussian_post_obj.entropy(loc=mu, log_scale=log_std)
    # Compare with the scipy implementation
    entropy_scipy = scipy.stats.norm.entropy(loc=mu, scale=std).sum()
    print(entropy_mine, entropy_scipy)
    assert np.allclose(entropy_mine, entropy_scipy)
    print("ok!")
