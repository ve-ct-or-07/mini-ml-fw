# create_best_model.py

from my_ml_lib.nn import Module, Sequential, Linear, ReLU


class MyMLP(Module):
    """
    Multi-Layer Perceptron (MLP) model architecture.

    Architecture used as the best model:
    [784 → 256 → 128 → 10]
    """

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.network = Sequential(
            Linear(n_features, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, n_classes)
        )

    def __call__(self, X):
        return self.network(X)

    def __repr__(self):
        # Delegate the representation to the internal Sequential network
        return f"{self.__class__.__name__}(\n{repr(self.network)}\n)"


def initialize_best_model():
    """
    This function MUST return an instance of your best model's architecture.
    It must match the architecture that was saved in 'best_model.npz'.
    """
    model = MyMLP(784, 10)  # 784 input features (e.g., flattened 28x28 images), 10 classes
    return model
