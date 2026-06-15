import numpy as np

class GaussianBasisFeatures:
    def __init__(self, n_centers=100, sigma=5.0, random_state=None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers_ = None
        self.random_state = random_state  # For reproducibility if using random sampling

    def fit(self, X, y=None):
        """
        Select n_centers random points from X as RBF centers.
        """
        # Strategy: Randomly sample n_centers points from X
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(X.shape[0], self.n_centers, replace=False)
        self.centers_ = X[indices]
        return self

    def transform(self, X):
        """
        Transform input X into Gaussian RBF features.
        Output shape: (n_samples, n_centers)
        """
        if self.centers_ is None:
            raise RuntimeError("Transformer is not fitted yet.")

        # Compute squared Euclidean distance between each sample and each center
        # Using broadcasting:
        # ||x - mu||^2 = sum((x_i - mu_i)^2) over features
        # X shape: (n_samples, n_features)
        # centers_ shape: (n_centers, n_features)
        sq_dists = np.sum((X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]) ** 2, axis=2)

        # Apply the Gaussian RBF formula: exp(-||x - mu||^2 / (2 * sigma^2))
        Phi = np.exp(-sq_dists / (2 * (self.sigma ** 2)))

        return Phi

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
