import numpy as np

class LeastSquaresClassifier:
    """
    Classifier using the least squares approach.

    Fits a linear model by minimizing the squared error between
    predictions and target labels (e.g., encoded as +1/-1 or one-hot).
    Prediction is typically based on the sign or argmax of the output.
    """
    def __init__(self):
        self.w_ = None # Weight vector (including bias)
        self.classes_ = None  # Store unique class labels for mapping

    def fit(self, X, y):
        """
        Fit the least squares classifier model.

        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels), shape (n_samples,).
        """
        # Encode labels
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Augment X with bias term
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])

        # Create target matrix T
        if n_classes == 2:
            # Binary classification → encode as +1/-1
            T = np.where(y == self.classes_[0], -1, 1).reshape(-1, 1)
        else:
            # Multi-class classification → one-hot encode
            T = np.zeros((y.shape[0], n_classes))
            for idx, cls in enumerate(self.classes_):
                T[y == cls, idx] = 1

        # Solve using pseudo-inverse for numerical stability
        self.w_ = np.linalg.pinv(X_aug) @ T
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # Augment X with bias column
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])

        # Compute scores
        scores = X_aug @ self.w_

        # Determine predictions
        if scores.ndim == 1 or scores.shape[1] == 1:
            # Binary classification: sign-based decision
            pred_inds = (scores.flatten() > 0).astype(int)
        else:
            # Multi-class classification: argmax-based decision
            pred_inds = np.argmax(scores, axis=1)

        predictions = self.classes_[pred_inds]

        return predictions
