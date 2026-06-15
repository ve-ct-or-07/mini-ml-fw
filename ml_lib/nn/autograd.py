# my_ml_lib/nn/autograd.py
import numpy as np
import math # For exp, log etc.

class Value:
    """
    Stores a scalar or numpy array and its gradient.
    Builds a computation graph for automatic differentiation (backpropagation).
    Inspired by micrograd: https://github.com/karpathy/micrograd
    """
    def __init__(self, data, _parents=(), _op='', label=''):
        """
        Initializes a Value object.

        Args:
            data: Input data (convertible to numpy float64 array).
            _parents (tuple): Parent Value objects that created this node.
            _op (str): Operation that created this node.
            label (str): Optional label for debugging/visualization.
        """
        # Data Type Conversion & Initialization ---
        # Ensure data is a numpy array for consistency later
        if not isinstance(data, np.ndarray):
            # Try converting lists/tuples/scalars to numpy arrays
            try:
                data = np.array(data, dtype=np.float64)
            except TypeError:
                raise TypeError(f"Data must be convertible to a numpy array, got {type(data)}")
        # Ensure data is float64 for precision
        if not np.issubdtype(data.dtype, np.floating):
             # print warning, then cast
             data = data.astype(np.float64)

        self.data = data
        # Initialize gradient with zeros, matching data shape
        self.grad = np.zeros_like(data, dtype=np.float64)
        # Internal variables for graph structure
        self._backward = lambda: None # Function to compute local gradients
        self._prev = set(_parents)   # Set of parent Value objects
        self._op = _op                # Operation that created this Value
        self.label = label            # Optional label for debugging/visualization


    def __repr__(self):
         # Provide a nice representation, showing shape for arrays
        data_str = f"array(shape={self.data.shape})" if self.data.ndim > 0 else f"scalar({self.data.item():.4f})"
        grad_str = f"array(shape={self.grad.shape})" if self.grad.ndim > 0 else f"scalar({self.grad.item():.4f})"
        return f"Value(data={data_str}, grad={grad_str}, op='{self._op}')"

    # --- Helper for un-broadcasting gradients back to original shape ---
    @staticmethod
    def _unbroadcast(grad, shape):
        """
        Sum-reduce `grad` to match `shape` (reverse numpy broadcasting).
        """
        # If shapes identical, nothing to do
        if grad.shape == shape:
            return grad

        # Sum out extra dimensions on the left
        while grad.ndim > len(shape):
            grad = np.sum(grad, axis=0)

        # Now handle singleton dimensions
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = np.sum(grad, axis=i, keepdims=True)
        return grad

    # --- TODO: Implement Core Operations ---
    ## Add operation with broadcasting has been given as an example 

    def __add__(self, other):
        # Ensure 'other' is also a Value object, wrap if necessary
        other = other if isinstance(other, Value) else Value(other)
        # Perform addition using numpy's broadcasting rules
        out_data = self.data + other.data
        out = Value(out_data, (self, other), '+')

        def _backward():
            # Gradient of '+' wrt self is 1, wrt other is 1.
            grad_self = out.grad
            grad_other = out.grad    

            # Undo broadcasting for self.grad
            if self.data.shape != grad_self.shape:
                grad_self = Value._unbroadcast(grad_self, self.data.shape)

            # Undo broadcasting for other.grad
            if other.data.shape != grad_other.shape:
                grad_other = Value._unbroadcast(grad_other, other.data.shape)

            # Accumulate gradients
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __mul__(self, other):
        """ Multiplication operation. Handles broadcasting. """
        other = other if isinstance(other, Value) else Value(other) # Ensure other is Value
        # --- Step 1 - Forward Pass ---
        out_data = self.data * other.data
        out = Value(out_data, (self, other), '*')

        # --- Step 2 - Define _backward for Multiplication ---
        def _backward():
            # Gradient of 'A * B' w.r.t A is B.data, w.r.t B is A.data.
            # Compute contributions
            grad = out.grad
            # w.r.t self:
            grad_self = grad * other.data
            # w.r.t other:
            grad_other = grad * self.data

            # Undo broadcasting
            if grad_self.shape != self.data.shape:
                grad_self = Value._unbroadcast(grad_self, self.data.shape)
            if grad_other.shape != other.data.shape:
                grad_other = Value._unbroadcast(grad_other, other.data.shape)

            # Accumulate gradients
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out
    
    # --- Make addition and multiplication commutative ---
    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other

    # --- Other necessary math operations ---
    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __truediv__(self, other): # self / other
        # Division is multiplication by the inverse (power of -1)
        return self * (other**-1)

    def __rtruediv__(self, other): # other / self
        return other * (self**-1)
    
    def __pow__(self, other):
        """ Power operation (only supports scalar power). """
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        # --- Step 1 - Forward Pass ---
        out_data = np.power(self.data, other)
        out = Value(out_data, (self,), f'**{other}')

        # --- Step 2 - Define _backward for Power ---
        def _backward():
            # d/dx x^n = n * x^(n-1)
            grad = out.grad * (other * np.power(self.data, other - 1))
            # Ensure shapes align
            if grad.shape != self.data.shape:
                grad = Value._unbroadcast(grad, self.data.shape)
            self.grad += grad

        out._backward = _backward
        return out

    # --- Activation Functions 
    def relu(self):
        """ Rectified Linear Unit (ReLU) activation. """
        # --- Forward Pass ---
        out_data = np.maximum(0, self.data)
        out = Value(out_data, (self,), 'ReLU')

        # --- Backward ---
        def _backward():
            grad = out.grad * (self.data > 0).astype(np.float64)
            if grad.shape != self.data.shape:
                grad = Value._unbroadcast(grad, self.data.shape)
            self.grad += grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """ Sigmoid activation: 1 / (1 + exp(-x)) """
        # Numerically stable implementation using clipping
        z = np.clip(self.data, -500, 500)
        out_data = 1.0 / (1.0 + np.exp(-z))
        out = Value(out_data, (self,), 'sigmoid')

        def _backward():
            s = out.data
            grad = out.grad * (s * (1.0 - s))
            if grad.shape != self.data.shape:
                grad = Value._unbroadcast(grad, self.data.shape)
            self.grad += grad

        out._backward = _backward
        return out
        
    def softmax(self, axis=-1):
        """Softmax activation along specified axis."""
        # --- Forward Pass ---
        # Subtract max for numerical stability
        z = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_z = np.exp(np.clip(z, -500, 500))
        sum_exp = np.sum(exp_z, axis=axis, keepdims=True)
        out_data = exp_z / sum_exp
        out = Value(out_data, (self,), 'softmax')

        # --- Backward Pass ---
        def _backward():
            # grad_out has same shape as out_data
            grad_out = out.grad
            p = out.data
            # For each sample, compute Jacobian-vector product efficiently
            grad = grad_out - np.sum(grad_out * p, axis=axis, keepdims=True)
            grad = grad * p
            if grad.shape != self.data.shape:
                grad = Value._unbroadcast(grad, self.data.shape)
            self.grad += grad

        out._backward = _backward
        return out

    # --- Elementary Functions (exp, log) ---
    def exp(self):
        """ Exponential function. """
        # --- Forward Pass ---
        clipped = np.clip(self.data, -500, 500)
        out_data = np.exp(clipped)
        out = Value(out_data, (self,), 'exp')

        # --- Backward ---
        def _backward():
            # derivative of exp is exp(self)
            grad = out.grad * out.data
            if grad.shape != self.data.shape:
                grad = Value._unbroadcast(grad, self.data.shape)
            self.grad += grad

        out._backward = _backward
        return out

    def log(self):
        """ Natural logarithm function (log base e). """
        eps = 1e-12
        safe = np.maximum(self.data, eps)
        out_data = np.log(safe)
        out = Value(out_data, (self,), 'log')

        def _backward():
            # d/dx log(x) = 1/x
            grad = out.grad / safe
            if grad.shape != self.data.shape:
                grad = Value._unbroadcast(grad, self.data.shape)
            self.grad += grad

        out._backward = _backward
        return out

    # --- Matrix Multiplication ---
    def __matmul__(self, other):
        """ Matrix multiplication (@ operator). """
        other = other if isinstance(other, Value) else Value(other) # Ensure other is Value
        out_data = self.data @ other.data
        out = Value(out_data, (self, other), '@')

        def _backward():
            # out = A @ B
            # dA = out.grad @ B.T
            # dB = A.T @ out.grad
            # Ensure correct broadcasting when necessary
            grad = out.grad
            # gradients for self
            if self.data.ndim == 2 and other.data.ndim == 2:
                grad_self = grad @ other.data.T
                grad_other = self.data.T @ grad
            else:
                # Fallback to numpy tensordot style for higher dims (best-effort)
                grad_self = np.matmul(grad, np.swapaxes(other.data, -1, -2))
                grad_other = np.matmul(np.swapaxes(self.data, -1, -2), grad)

            if grad_self.shape != self.data.shape:
                grad_self = Value._unbroadcast(grad_self, self.data.shape)
            if grad_other.shape != other.data.shape:
                grad_other = Value._unbroadcast(grad_other, other.data.shape)

            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    # --- Reduction Operations (sum, mean) ---
    def sum(self, axis=None, keepdims=False):
        """ Summation operation. """
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Value(out_data, (self,), 'sum')

        def _backward():
            grad = out.grad
            # If axis specified and keepdims=False, we must reshape grad to have singleton dims
            if axis is not None and not keepdims:
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = tuple(axis)
                # Normalize negative axes
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                shape = list(self.data.shape)
                for a in axes:
                    shape[a] = 1
                grad = grad.reshape(shape)
            # Broadcast grad to self.data.shape
            grad_broadcast = np.broadcast_to(grad, self.data.shape)
            self.grad += grad_broadcast

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """ Mean operation. """
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Value(out_data, (self,), 'mean')

        def _backward():
            # Number of elements averaged over
            if axis is None:
                N = self.data.size
                grad = out.grad / N
                grad = np.broadcast_to(grad, self.data.shape)
            else:
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = tuple(axis)
                # Normalize negative axes
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                # compute N as product of dims along axes
                N = 1
                for a in axes:
                    N *= self.data.shape[a]
                grad = out.grad
                if not keepdims:
                    # reshape grad to have singleton dims at reduced axes
                    shape = list(self.data.shape)
                    for a in axes:
                        shape[a] = 1
                    grad = grad.reshape(shape)
                grad = np.broadcast_to(grad, self.data.shape)
                grad = grad / N
            self.grad += grad

        out._backward = _backward
        return out

    #  BACKPROPAGATION 
    def backward(self):
        """ Performs backpropagation starting from this Value node. """
        # --- Step 1 - Topological Sort ---
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # --- Step 2 - Initialize Gradient ---
        # Seed gradient of final node with ones (same shape)
        self.grad = np.ones_like(self.data, dtype=np.float64)

        # --- Step 3 - Backward Pass ---
        for node in reversed(topo):
            node._backward()
