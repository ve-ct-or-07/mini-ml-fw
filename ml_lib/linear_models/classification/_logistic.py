import numpy as np

class LogisticRegression:
    """
    L2-Regularized Logistic Regression classifier using SGD optimization.
    """
    def __init__(self, alpha=0.01, epochs=50, lr=0.01, batch_size=64, fit_intercept=True, random_state=None):
        """
        Args:
            alpha (float): L2 regularization strength.
            epochs (int): Number of epochs for SGD.
            lr (float): Learning rate for SGD.
            batch_size (int): Mini-batch size.
            fit_intercept (bool): Whether to add a bias term.
            random_state (int, optional): Random seed.
        """
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.w_ = None

    def _sigmoid(self, z):
        """Numerically stable sigmoid function."""
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X, y):
        """
        Fit the logistic regression model using stochastic gradient descent (SGD).

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values (0 or 1), shape (n_samples,).
        """
        y = y.astype(float)
        n_samples, n_features = X.shape

        # Handle intercept and weight initialization
        if self.fit_intercept:
            X_aug = np.hstack((np.ones((n_samples, 1)), X))
            self.w_ = np.zeros(n_features + 1)
        else:
            X_aug = X
            self.w_ = np.zeros(n_features)

        # Set seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # SGD loop
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X_aug[indices], y[indices]

            # Mini-batch iteration
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Predictions
                h = self._sigmoid(X_batch @ self.w_)

                # Gradient (BCE + L2 regularization)
                gradient = (X_batch.T @ (h - y_batch)) / y_batch.size

                if self.fit_intercept:
                    gradient[1:] += (self.alpha / y_batch.size) * self.w_[1:]
                else:
                    gradient += (self.alpha / y_batch.size) * self.w_

                # Weight update
                self.w_ -= self.lr * gradient

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities [P(y=0|X), P(y=1|X)], shape (n_samples, 2).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        if self.fit_intercept:
            X_aug = np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            X_aug = X

        z = X_aug @ self.w_
        prob_y1 = self._sigmoid(z)
        prob_y0 = 1 - prob_y1
        return np.column_stack([prob_y0, prob_y1])

    def predict(self, X):
        """
        Predict class labels (0 or 1) for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (0 or 1), shape (n_samples,).
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)
