# my_ml_lib/nn/optim.py
import numpy as np
from .autograd import Value # Need Value to check parameter type

class SGD:
    """
    Implements stochastic gradient descent.
    """
    def __init__(self, params, lr=0.01):
        """
        Initializes the SGD optimizer.

        Args:
            params (iterable): Iterable of Value objects (typically model.parameters()).
                               It's recommended to convert generators to lists.
            lr (float): Learning rate.
        """
        # Ensure params is stored as a list
        self.params = list(params)
        self.lr = lr

        # Validate parameters and initialize gradients if needed (good practice)
        for p in self.params:
            if not isinstance(p, Value):
                 raise TypeError("Optimizer parameters must be Value objects.")
            # Initialize grad attribute if it doesn't exist or is None
            if not hasattr(p, 'grad') or p.grad is None:
                 p.grad = np.zeros_like(p.data, dtype=np.float64)

    def step(self):
        """
        Performs a single optimization step (parameter update).
        Updates the .data attribute of each parameter based on its .grad.
        """
        for p in self.params:
            if p is None:
                continue
            if not isinstance(p, Value):
                raise TypeError("Optimizer parameters must be Value objects.")
            if p.grad is None:
                continue
            # Update in-place
            p.data[:] = p.data - self.lr * p.grad

    def zero_grad(self):
        """
        Sets the gradients (.grad attribute) of all managed parameters to zero.
        It's crucial to call this before computing gradients for a new batch (i.e., before loss.backward()).
        """
        for p in self.params:
            if p is None:
                continue
            if not hasattr(p, 'grad') or p.grad is None:
                p.grad = np.zeros_like(p.data, dtype=np.float64)
            else:
                p.grad[:] = 0.0

    def __repr__(self):
        """Provides a string representation of the optimizer."""
        return f"SGD(lr={self.lr})"
