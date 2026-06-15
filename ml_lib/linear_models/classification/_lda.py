import numpy as np

class LinearDiscriminantAnalysis:
    """
    Linear Discriminant Analysis (LDA) classifier using the least squares approach.

    Fits a linear regression model to predict target values derived from class labels.
    """
    def __init__(self):
        self.w_ = None  # Projection vector (including bias term)
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the LDA model using the least squares target encoding.

        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels, 0 or 1), shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("This LDA implementation currently supports only binary classification.")
        
        # --- Augment X with a bias column ---
        X_augmented = np.hstack([np.ones((n_samples, 1)), X])

        # --- Create target vector T based on class membership ---
        # Target for class 1: N / N_1
        # Target for class 0: -N / N_0
        N = n_samples
        N_1 = np.sum(y == self.classes_[1])  # Count of class 1
        N_0 = np.sum(y == self.classes_[0])  # Count of class 0

        if N_1 == 0 or N_0 == 0:
            raise ValueError("Training data must contain samples from both classes.")

        T = np.where(y == self.classes_[1], N / N_1, -N / N_0)

        # --- Calculate weights using pseudo-inverse ---
        # w = (X_aug^T X_aug)^(-1) X_aug^T T  =>  w = pinv(X_aug) @ T
        try:
            # Use pseudo-inverse for numerical stability
            self.w_ = np.linalg.pinv(X_augmented) @ T
            # Alternative: explicit normal equation (less stable if X^T X is singular)
            # self.w_ = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ T
        except np.linalg.LinAlgError:
            print("Error: Could not compute weights using pseudo-inverse. Matrix might be ill-conditioned.")
            self.w_ = None # Indicate fitting failed
            raise # Re-raise the error

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (0 or 1), shape (n_samples,).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # --- Augment X with a bias column ---
        X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])

        # --- Calculate the linear scores ---
        scores = X_augmented @ self.w_

        # --- Apply threshold and map to original class labels ---
        # Threshold is typically 0 for this target encoding
        # Assign to class 1 if score > 0, otherwise class 0
        predictions = np.where(scores > 0, self.classes_[1], self.classes_[0])

        return predictions

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)
