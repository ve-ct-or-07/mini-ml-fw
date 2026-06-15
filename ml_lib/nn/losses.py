# my_ml_lib/nn/losses.py

import numpy as np
from .modules.base import Module
from .autograd import Value


### You would use this for one vs rest Logistic Regression
class BinaryCrossEntropyLoss(Module):
    """
    Computes the Binary Cross Entropy loss between logits and targets (0 or 1).
    Assumes the input is a single logit (pre-sigmoid score) per sample.
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def __call__(self, logits: Value, targets: np.ndarray) -> Value:
        """
        Calculates the Binary Cross Entropy loss.

        Args:
            logits (Value): Input logits (pre-sigmoid scores), shape (batch_size,) or (batch_size, 1).
            targets (np.ndarray): Ground truth labels (0 or 1), shape (batch_size,).

        Returns:
            Value: The computed loss (scalar if reduction is 'mean' or 'sum').
        """
        # Ensure targets are float64 and have the correct shape for broadcasting
        targets_np = targets.astype(np.float64).reshape(-1, 1) # Ensure column vector

        # Step 1 - Apply Sigmoid
        probs = logits.sigmoid()

        # Step 2 - Numerical Stability (Value.log handles stability via eps)
        # Step 3 - Wrap Targets
        targets_val = Value(targets_np)

        # Step 4 - Calculate BCE Formula
        # BCE = - [ y * log(p) + (1-y) * log(1-p) ]
        # Use Value operations so that autograd can trace them
        # Note: (1 - probs) is also a Value
        loss_elements = - (targets_val * probs.log() + (Value(1.0) - targets_val) * (Value(1.0) - probs).log())  # shape (batch, 1)

        # Step 5 - Apply Reduction
        if self.reduction == 'mean':
            return loss_elements.mean()
        elif self.reduction == 'sum':
            return loss_elements.sum()
        else:  # none
            return loss_elements

    def __repr__(self):
        """String representation."""
        return f"BinaryCrossEntropyLoss(reduction='{self.reduction}')"


class CrossEntropyLoss(Module):
    """
    Computes the cross-entropy loss between input logits and target class indices.
    Combines LogSoftmax and NLLLoss.
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction
        # No internal storage needed if using pure autograd approach

    def __call__(self, input_logits: Value, target: np.ndarray) -> Value:
        """
        Computes the cross-entropy loss using only Value operations.

        Args:
            input_logits (Value): Raw logits, shape (batch_size, n_classes).
            target (np.ndarray): Ground truth labels (class indices 0 to n_classes-1), shape (batch_size,).

        Returns:
            Value: The computed loss.
        """
        batch_size, n_classes = input_logits.data.shape

        # Step 1 - Calculate LogSoftmax (numerically stable)
        # a) max per sample (numpy)
        max_logits = np.max(input_logits.data, axis=1, keepdims=True)
        stable = input_logits - Value(max_logits)               # stable logits (Value)
        exp_stable = stable.exp()                               # Value
        sum_exp = exp_stable.sum(axis=1, keepdims=True)         # Value
        log_sum_exp = sum_exp.log()                             # Value
        log_probs = stable - log_sum_exp                        # Value (batch, n_classes)

        # Step 2 - Create One-Hot Targets (numpy)
        y_one_hot_np = np.zeros((batch_size, n_classes), dtype=np.float64)
        y_one_hot_np[np.arange(batch_size), target] = 1.0

        # Step 3 - Wrap One-Hot Targets
        y_one_hot = Value(y_one_hot_np)

        # Step 4 - Calculate NLL
        nll = - (y_one_hot * log_probs).sum(axis=1)  # shape (batch_size,)
        # note: sum(axis=1) yields shape (batch_size,)

        # Step 5 - Apply Reduction
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll

    def __repr__(self):
        """String representation."""
        return f"CrossEntropyLoss(reduction='{self.reduction}')"
