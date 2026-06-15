import numpy as np

class BayesianRegression:
    """
    Bayesian Linear Regression with Gaussian basis functions.

    Assumes a Gaussian likelihood and a Gaussian prior on weights.
    Computes the posterior distribution over weights and the
    posterior predictive distribution. (Based on A2 logic).
    """
    def __init__(self, n_basis=25, basis_sigma_fraction=0.1, alpha=1.0, beta=100.0):
        """
        Args:
            n_basis (int): Number of Gaussian basis functions (including bias).
            basis_sigma_fraction (float): Width sigma as fraction of center spacing.
            alpha (float): Precision of the Gaussian prior on weights (1/variance).
            beta (float): Precision of the Gaussian likelihood noise (1/variance).
        """
        self.n_basis = n_basis
        self.basis_sigma_fraction = basis_sigma_fraction
        self.alpha = alpha
        self.beta = beta

        self.basis_centers_ = None
        self.basis_sigma_ = None

        self.posterior_mean_ = None  # m_N
        self.posterior_cov_ = None   # S_N

    def _gaussian_basis(self, X):
        """Transforms input X using Gaussian basis functions."""
        if self.basis_centers_ is None or self.basis_sigma_ is None:
            raise RuntimeError("Basis functions parameters not set. Call fit first.")

        diff = X - self.basis_centers_.T # X is (N,1), centers is (M-1,1), so centers.T is (1,M-1)
        sq_dist = diff**2
        phi = np.exp(-sq_dist / (2 * self.basis_sigma_**2))
        
        # Add the bias term phi_0 = 1
        return np.hstack([np.ones((X.shape[0], 1)), phi])

    def fit(self, X, y):
        """
        Compute the posterior distribution over weights.

        Args:
            X (np.ndarray): Training input vectors, shape (n_samples, n_features=1).
                            (Assuming 1D input based on A2 sine example).
            y (np.ndarray): Target values, shape (n_samples,).
        """

        # 1. Determine basis function centers (mu_j) and width (sigma)
        if self.n_basis > 1:
            # Equidistant centers from min to max of X
            self.basis_centers_ = np.linspace(X.min(), X.max(), self.n_basis - 1).reshape(-1, 1)
            # Sigma as a fraction of the distance between centers
            spacing = (X.max() - X.min()) / (self.n_basis - 2) if self.n_basis > 2 else 1.0
            self.basis_sigma_ = self.basis_sigma_fraction * spacing
        else: # Only a bias term
            self.basis_centers_ = np.array([])
            self.basis_sigma_ = 1.0
        
        y = y.flatten()

        # 2. Transform X using basis functions
        Phi = self._gaussian_basis(X)

        # 3. Posterior covariance: S_N = inv(alpha * I + beta * Phi^T @ Phi)
        S_N_inv = self.alpha * np.eye(self.n_basis) + self.beta * (Phi.T @ Phi)
        self.posterior_cov_ = np.linalg.inv(S_N_inv)

        # 4. Posterior mean: m_N = beta * S_N @ Phi^T @ y
        self.posterior_mean_ = self.beta * self.posterior_cov_ @ Phi.T @ y

        return self

    def predict_dist(self, X):
        """
        Compute the posterior predictive distribution for new inputs X.

        Args:
            X (np.ndarray): New input vectors, shape (n_samples, n_features=1).

        Returns:
            tuple: (predictive_mean, predictive_variance) numpy arrays, both shape (n_samples,).
                   predictive_variance is s^2(x), not std dev s(x).
        """
        if self.posterior_mean_ is None or self.posterior_cov_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # 1. Transform X using basis functions
        Phi_new = self._gaussian_basis(X)

        # 2. Predictive mean
        pred_mean = Phi_new @ self.posterior_mean_

        # 3. Predictive variance: 1/beta + diag(Phi_new @ S_N @ Phi_new^T)
        pred_var = 1.0 / self.beta + np.sum((Phi_new @ self.posterior_cov_) * Phi_new, axis=1)

        return pred_mean, pred_var

    def predict(self, X):
        """
        Predict target values using the mean of the posterior predictive distribution.

        Args:
            X (np.ndarray): New input vectors, shape (n_samples, n_features=1).

        Returns:
            np.ndarray: Predicted values (mean of predictive distribution), shape (n_samples,).
        """
        pred_mean, _ = self.predict_dist(X)
        return pred_mean
