# lasso.py

import numpy as np

class LassoRegressor:
    """
    Simple Lasso regression via proximal gradient descent.

    Parameters
    ----------
    alpha : float
        Regularization strength.
    learning_rate : float
        Step size for gradient descent.
    max_iter : int
        Max number of iterations.
    tol : float
        Tolerance for convergence.
    verbose : bool
        If True, print progress.
    """

    def __init__(self, alpha=0.1, learning_rate=0.01, max_iter=1000, tol=1e-6, verbose=True):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.coef_ = None
        self.intercept_ = 0.0
        self.train_mse_history = []

    def _soft_threshold(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0
        self.train_mse_history = []

        prev_coef = self.coef_.copy()

        for iteration in range(self.max_iter):
            y_pred = X @ self.coef_ + self.intercept_
            error = y_pred - y
            mse = np.mean(error ** 2)
            self.train_mse_history.append(mse)

            grad_w = (1.0 / n_samples) * X.T @ error
            grad_b = np.mean(error)

            coef_temp = self.coef_ - self.learning_rate * grad_w
            self.coef_ = self._soft_threshold(coef_temp, self.learning_rate * self.alpha)
            self.intercept_ -= self.learning_rate * grad_b

            delta = np.linalg.norm(self.coef_ - prev_coef)
            if delta < self.tol:
                if self.verbose:
                    print(f"[INFO] Converged at iteration {iteration} (Δβ = {delta:.2e})")
                break

            prev_coef = self.coef_.copy()

            if self.verbose and (iteration % 10 == 0):
                print(f"[ITER {iteration:4d}] MSE: {mse:.6f}")

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def get_mse_history(self):
        return self.train_mse_history

    def get_coefficients(self):
        return self.coef_

    def get_intercept(self):
        return self.intercept_
