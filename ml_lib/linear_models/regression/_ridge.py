import numpy as np

class RidgeRegression:
    """
    Linear least squares with L2 regularization (Ridge Regression).

    Minimizes objective function: ||y - Xw||^2 + alpha * ||w||^2
    """
    def __init__(self, alpha=1.0, fit_intercept=True):
        """
        Args:
            alpha (float): Regularization strength; must be a positive float.
                           Larger values specify stronger regularization.
            fit_intercept (bool): Whether to calculate the intercept for this model.
                                  If set to False, no intercept will be used.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None  # Weights for the features
        self.intercept_ = None  # Intercept (bias term)

    def fit(self, X, y):
        """
        Fit Ridge regression model.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values, shape (n_samples,).
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)  # ensure column vector
        n_samples, n_features = X.shape

        # 1. Add intercept term if needed
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_aug = X

        # 2. Create regularization matrix
        n_params = X_aug.shape[1]
        I = np.eye(n_params)

        # Do not regularize the intercept term
        if self.fit_intercept:
            I[0, 0] = 0.0

        # 3. Solve normal equation: w = (X^T X + alpha * I)^(-1) X^T y
        A = X_aug.T @ X_aug + self.alpha * I
        b = X_aug.T @ y
        w = np.linalg.solve(A, b)

        # 4. Extract coefficients
        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = w.flatten()

        return self

    def predict(self, X):
        """
        Predict using the linear model.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values, shape (n_samples,).
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")

        # Prediction: y_pred = Xw + b
        y_pred = X @ self.coef_ + self.intercept_
        return y_pred
