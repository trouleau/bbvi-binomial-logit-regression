import pandas as pd
import numpy as np
import torch

from bbvi.models import ModelBinomialLogit
from bbvi.learners import Learner, CallbackMonitor


if __name__ == "__main__":

    # np.random.seed(42)

    # Generate synthetic toy example data
    # ===================================
    print("Generate toy data...")
    n_points = 1000
    n_covariates = 10
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

    print("Init LearnerMLE...")
    callback = CallbackMonitor(print_every=100)
    learner = Learner(
        model,
        lr=1.0,
        lr_gamma=1.0,
        tol=1e-6,
        max_iter=200000
    )
    print("fit model with MLE learner...")
    x0 = torch.tensor(np.random.randn(n_covariates), dtype=torch.float64)
    print("x0:")
    print(x0)
    beta_hat = learner.fit(covariate_df, response_df, x0, callback=callback)
    beta_hat = beta_hat.detach().numpy()
    print(f"- True beta:")
    print(beta)
    print(f"- Maximum loglikelihood attained at beta_hat:")
    print(beta_hat)
    mae = np.abs(beta-beta_hat).sum()
    print(f"- MAE: {mae}")
