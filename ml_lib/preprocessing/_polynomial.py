import numpy as np
from itertools import combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None
        self.powers_ = None  # Stores exponents for each output feature

    def fit(self, X, y=None):
        n_features = X.shape[1]
        self.n_input_features_ = n_features

        # Generate all combinations of feature indices with replacement
        combs = []
        for d in range(0 if self.include_bias else 1, self.degree + 1):
            combs.extend(combinations_with_replacement(range(n_features), d))
        self.powers_ = list(combs)
        self.n_output_features_ = len(self.powers_)

        return self

    def transform(self, X):
        if self.powers_ is None:
            raise RuntimeError("PolynomialFeatures instance is not fitted yet. Call 'fit' first.")

        n_samples = X.shape[0]
        X_poly = np.empty((n_samples, self.n_output_features_), dtype=float)

        for i, comb in enumerate(self.powers_):
            # Multiply the corresponding feature columns together
            X_poly[:, i] = np.prod(X[:, comb], axis=1)

        return X_poly

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
