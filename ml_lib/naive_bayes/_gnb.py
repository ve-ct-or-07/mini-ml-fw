import numpy as np

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes (GNB) classifier.

    Assumes features are conditionally independent given the class,
    and each feature follows a Gaussian distribution within each class.
    """
    def __init__(self):
        self.classes_ = None
        self.class_priors_ = None  # P(y=k)
        self.theta_ = None         # Mean of each feature per class
        self.var_ = None           # Variance of each feature per class
        self.epsilon_ = 1e-9       # To prevent division by zero in variance

    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes according to X, y.

        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels), shape (n_samples,).
        """
        # --- Implemented GNB fitting logic ---
        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Class priors: P(y=k)
        self.class_priors_ = counts / X.shape[0]

        # Means (theta_) and variances (var_)
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = np.mean(X_c, axis=0)
            self.var_[idx, :] = np.var(X_c, axis=0) + self.epsilon_  # Add epsilon for stability

        return self

    def _gaussian_log_pdf(self, X, class_idx):
        """Calculate log probability density function for Gaussian."""
        mean = self.theta_[class_idx]
        var = self.var_[class_idx]

        # --- Implement Gaussian log PDF calculation ---
        # log P(x_j | y=k) = -0.5 * [log(2πσ²) + ((x - μ)² / σ²)]
        log_prob = -0.5 * np.sum(
            np.log(2. * np.pi * var) + ((X - mean) ** 2) / var,
            axis=1
        )
        return log_prob  # Shape (n_samples,)

    def predict_log_proba(self, X):
        """
        Calculate log probability estimates for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Log probability of samples for each class, shape (n_samples, n_classes).
        """
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        joint_log_likelihood = np.zeros((n_samples, n_classes))

        # --- Implement log probability prediction ---
        for i in range(n_classes):
            class_conditional_log_prob = self._gaussian_log_pdf(X, i)
            joint_log_likelihood[:, i] = np.log(self.class_priors_[i]) + class_conditional_log_prob

        return joint_log_likelihood

    def predict_proba(self, X):
        """
        Calculate probability estimates for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability of samples for each class, shape (n_samples, n_classes).
        """
        log_proba = self.predict_log_proba(X)
        # --- Stable softmax for numerical stability ---
        max_log_proba = np.max(log_proba, axis=1, keepdims=True)
        exp_log_proba = np.exp(log_proba - max_log_proba)
        proba = exp_log_proba / np.sum(exp_log_proba, axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """
        Perform classification on samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        log_proba = self.predict_log_proba(X)
        preds = self.classes_[np.argmax(log_proba, axis=1)]
        return preds
