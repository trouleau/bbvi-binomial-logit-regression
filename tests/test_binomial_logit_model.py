import pandas as pd
import numpy as np
import torch

from bbvi.models import ModelBinomialLogit


if __name__ == "__main__":

    # np.random.seed(42)

    # Generate synthetic toy example data
    # ===================================
    print("Generate toy data...")
    n_points = 10000
    n_covariates = 1
    print(f"- Generate {n_points} data points with {n_covariates} covariates")
    # Define the convariates
    X = np.random.normal(1.0, 1.0, size=(n_points, n_covariates))
    # Define the counts
    m = np.random.randint(30, 60, size=n_points)
    # Define the parameters
    beta = np.random.normal(loc=-1.0, scale=5.0, size=n_covariates)
    print("- True beta:", beta)
    # Transform into probability
    prob = 1 / (1 + np.exp(-X.dot(beta)))
    # Sample responses
    y = np.random.binomial(m, prob)
    # Format as dataframe
    covariate_df = pd.DataFrame(
        data=X, columns=['col_{k}' for k in range(n_covariates)])
    response_df = pd.DataFrame(index=covariate_df.index)
    response_df['total_count'] = m
    response_df['positive_count'] = y

    # Init and fit the model
    # ======================
    print("Init ModelBinomialLogit...")
    model = ModelBinomialLogit()

    print("Set data...")
    model.set_data(covariate_df, response_df)

    print("Evaluate log-likelihood...")

    n_steps = 10000
    beta_hat_arr = np.linspace(beta[0]-1.0, beta[0]+1.0, n_steps)
    loglik_arr = np.zeros_like(beta_hat_arr)
    for i, x in enumerate(beta_hat_arr):
        beta_t = torch.tensor([x], dtype=torch.float64)
        value = model.log_likelihood(beta_t)
        loglik_arr[i] = value
        print(f"\r    {i:>5d}/{n_steps:>5d} | {x:.4f} | {value:.4f}", end="")
    print()
    best_beta = beta_hat_arr[np.argmax(loglik_arr)]
    print(f"- True beta: {beta[0]:.6f}")
    print(f"- Maximum loglikelihood attained at beta_hat: {best_beta:.6f}")
    print(f"- Difference abs(beta_true - beta_hat) = {abs(beta[0]-best_beta)}")
