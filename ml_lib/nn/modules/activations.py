from .base import Module

class ReLU(Module):
    """Applies the Rectified Linear Unit function element-wise."""
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        """
        Forward pass: Applies ReLU activation.

        Args:
            x (Value): Input Value object.

        Returns:
            Value: Output Value object (result of x.relu()).
        """
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Applies the Sigmoid function element-wise."""
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        """
        Forward pass: Applies Sigmoid activation.

        Args:
            x (Value): Input Value object.

        Returns:
            Value: Output Value object (result of x.sigmoid()).
        """
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"


class Softmax(Module):
    """Applies the Softmax function along a specified axis."""
    def __init__(self, axis=-1):
        """
        Args:
            axis (int): Axis along which Softmax is computed. Default is -1 (last axis).
        """
        super().__init__()
        self.axis = axis

    def __call__(self, x):
        """
        Forward pass: Applies Softmax activation.

        Args:
            x (Value): Input Value object.

        Returns:
            Value: Output Value object (result of x.softmax(axis=self.axis)).
        """
        return x.softmax(axis=self.axis)

    def __repr__(self):
        return f"Softmax(axis={self.axis})"
