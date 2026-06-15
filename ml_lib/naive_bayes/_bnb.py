import numpy as np

class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes (BNB) classifier.

    Assumes features are binary (0 or 1), conditionally independent
    given the class, and follow a Bernoulli distribution within each class.
    Often used for text classification with presence/absence features.
    """
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Additive (Laplace/Lidstone) smoothing parameter
                           (0 for no smoothing). Corresponds to Beta(alpha, alpha) prior.
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None  # Log P(y=k)
        self.feature_log_prob_ = None  # Log P(x_j=1 | y=k), shape (n_classes, n_features)

    def fit(self, X, y):
        """
        Fit Bernoulli Naive Bayes according to X, y.

        Assumes X contains binary features (0 or 1).

        Args:
            X (np.ndarray): Training vectors (binary), shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels), shape (n_samples,).
        """
        # --- Implemented BNB fitting logic ---
        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Class log priors: log(P(y=k))
        self.class_log_prior_ = np.log(counts / X.shape[0])

        # Conditional probabilities with Laplace smoothing
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            N_k = X_c.shape[0]
            # Count of feature=1 given class
            N_kj = np.sum(X_c, axis=0)
            # Apply Laplace smoothing
            prob = (N_kj + self.alpha) / (N_k + 2 * self.alpha)
            # Store log-probabilities
            self.feature_log_prob_[idx, :] = np.log(prob)

        return self

    def predict_log_proba(self, X):
        """
        Calculate log probability estimates for samples in X.

        Assumes X contains binary features (0 or 1).

        Args:
            X (np.ndarray): Samples (binary), shape (n_samples, n_features).

        Returns:
            np.ndarray: Log probability of samples for each class, shape (n_samples, n_classes).
        """
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")

        log_prob_x1 = self.feature_log_prob_
        # log(1 - exp(log_prob_x1)) needs numerical stability
        log_prob_x0 = np.log(1 - np.exp(log_prob_x1) + 1e-9)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        joint_log_likelihood = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            log_p_x_given_k = X * log_prob_x1[i] + (1 - X) * log_prob_x0[i]
            sum_log_p = np.sum(log_p_x_given_k, axis=1)
            joint_log_likelihood[:, i] = self.class_log_prior_[i] + sum_log_p

        return joint_log_likelihood

    def predict_proba(self, X):
        """
        Calculate probability estimates for samples in X.

        Args:
            X (np.ndarray): Samples (binary), shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability of samples for each class, shape (n_samples, n_classes).
        """
        log_proba = self.predict_log_proba(X)
        # Stable softmax implementation
        max_log = np.max(log_proba, axis=1, keepdims=True)
        exp_log = np.exp(log_proba - max_log)
        proba = exp_log / np.sum(exp_log, axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """
        Perform classification on samples in X.

        Args:
            X (np.ndarray): Samples (binary), shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        log_proba = self.predict_log_proba(X)
        # Argmax to get most likely class
        preds = self.classes_[np.argmax(log_proba, axis=1)]
        return preds
