import numpy as np

class Perceptron:
    """
    Perceptron classifier.

    Simple algorithm for binary linear classification.
    Uses iterative updates based on misclassified points.
    """
    def __init__(self, learning_rate=0.01, max_iters=1000, random_state=None):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.random_state = random_state
        self.w_ = None  # Weights (including bias)
        self.errors_ = []  # To store number of misclassifications per epoch

    def fit(self, X, y):
        """
        Fit perceptron model.

        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels, e.g., 0/1 or -1/1), shape (n_samples,).
        """
        # Initialize random generator
        rng = np.random.RandomState(self.random_state)

        # 1. Initialize weights (including bias)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        # 2. Augment X with bias column
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])

        # 3. Ensure y labels are +1 / -1
        y_unique = np.unique(y)
        if set(y_unique) == {0, 1}:
            y_mod = np.where(y == 0, -1, 1)
        elif set(y_unique) == {-1, 1}:
            y_mod = y.copy()
        else:
            raise ValueError("y must contain binary labels {0,1} or {-1,1}")

        self.errors_ = []

        # Pocket variables
        best_w = self.w_.copy()
        best_error = np.inf

        # 4. Perceptron learning iterations
        for _ in range(self.max_iters):
            errors = 0

            for i in range(X.shape[0]):
                # Compute prediction
                score = np.dot(X_aug[i], self.w_)
                y_pred = 1 if score >= 0 else -1

                # If misclassified, update weights
                if y_pred != y[i]:
                    self.w_ += self.learning_rate * (y[i] - y_pred) * X_aug[i]
                    errors += 1

            # Track misclassifications
            self.errors_.append(errors)

            # Pocket algorithm: keep best weights (lowest error)
            if errors < best_error:
                best_error = errors
                best_w = self.w_.copy()

            # Early stopping if perfect classification
            if errors == 0:
                break

        # Assign best weights
        self.w_ = best_w
        return self

    def _predict_raw(self, X):
        """Calculate net input (scores)."""
        X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_augmented @ self.w_

    def predict(self, X):
        """
        Return class label after unit step.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (e.g., 0/1 or -1/1 depending on fit).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # Apply activation (step function)
        scores = self._predict_raw(X)
        predictions = np.where(scores >= 0.0, 1, -1)

        return predictions
