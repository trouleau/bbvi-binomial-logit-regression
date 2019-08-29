import torch


class CallbackMonitor:

    def __init__(self, print_every=10):
        self.print_every = print_every

    def __call__(self, learner_obj, end='', force=False):
        t = learner_obj._n_iter_done + 1
        if force or (t % self.print_every == 0):
            dx = torch.abs(learner_obj.coeffs - learner_obj.coeffs_prev).max()
            print("\r    "
                  f"iter: {t:>4d}/{learner_obj.max_iter:>4d} | "
                  f"loss: {learner_obj.loss:.4f} | "
                  f"dx: {dx:.2e}"
                  "    ", end=end, flush=True)
        # if t % 1000 == 0:
        #     print()
        #     print('ll_value:', learner_obj.model.debug['ll_value'].item())
        #     print('prior_value:', learner_obj.model.debug['prior_value'].item())
        #     print('entropy_value:', learner_obj.model.debug['entropy_value'].item())


class Learner:

    def __init__(self, model, lr, lr_gamma, tol, max_iter):
        self.model = model
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.tol = tol
        self.max_iter = max_iter

    def _set_data(self, covariate_df, response_df):
        self.model.set_data(covariate_df, response_df)

    def _check_convergence(self):
        if torch.abs(self.coeffs - self.coeffs_prev).max() < self.tol:
            return True
        return False

    def fit(self, covariate_df, response_df, x0, callback=None):
        self._set_data(covariate_df, response_df)
        self.coeffs = x0.clone().detach().requires_grad_(True)
        self.coeffs_prev = self.coeffs.detach().clone()
        self.optimizer = torch.optim.Adam([self.coeffs], lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.lr_gamma)
        for t in range(self.max_iter):
            self._n_iter_done = t
            # Gradient step
            self.optimizer.zero_grad()
            self.loss = self.model.objective(self.coeffs)
            self.loss.backward()
            self.optimizer.step()
            if torch.isnan(self.coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')
            # Convergence check
            if self._check_convergence():
                break
            elif callback:  # Callback at each iteration
                callback(self, end='')
            self.coeffs_prev = self.coeffs.detach().clone()
        if callback:  # Callback before the end
            callback(self, end='\n', force=True)
        return self.coeffs
