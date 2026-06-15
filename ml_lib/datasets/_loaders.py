import numpy as np
import pandas as pd
import os

class DatasetNotFoundError(Exception):
    """Custom exception for when a dataset file is not found."""
    pass

def load_spambase(data_folder="data", filename="spambase.data", download_url=None):
    """
    Loads the UCI Spambase dataset from a local file.

    Args:
        data_folder (str): The folder where dataset files are stored.
        filename (str): The name of the spambase data file.
        download_url (str, optional): URL to download from if file not found.
                                       (Implementation of download is optional).

    Returns:
        tuple: (X, y) numpy arrays, features and labels.

    Raises:
        DatasetNotFoundError: If the dataset file cannot be found.
    """
    file_path = os.path.join(data_folder, filename)
    if not os.path.exists(file_path):
        # Optional: Add code here to download from download_url if provided
        raise DatasetNotFoundError(
            f"Dataset file not found at {file_path}. "
            f"Please download it from the UCI ML Repository and place it in the '{data_folder}' directory."
        )

    # The spambase data has no header and is comma-separated
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    
    return X, y

def load_fashion_mnist(data_folder="data", train_filename="fashion-mnist_train.csv",
                       test_filename="fashion-mnist_test.csv", kind='train', normalize=True):
    """
    Loads the Fashion-MNIST dataset from local CSV files.

    Args:
        data_folder (str): Folder where dataset CSV files are stored.
        train_filename (str): Name of the training CSV file.
        test_filename (str): Name of the testing CSV file.
        kind (str): 'train' or 'test' to specify which dataset to load.
        normalize (bool): If True, scale pixel values from 0-255 to 0-1.

    Returns:
        tuple: (X, y) numpy arrays, features (images flattened) and labels.

    Raises:
        DatasetNotFoundError: If the specified dataset file cannot be found.
        ValueError: If kind is not 'train' or 'test'.
    """
    if kind == 'train':
        filename = train_filename
    elif kind == 'test':
        filename = test_filename
    else:
        raise ValueError("kind must be 'train' or 'test'")

    file_path = os.path.join(data_folder, filename)
    if not os.path.exists(file_path):
        raise DatasetNotFoundError(
            f"Dataset file not found at {file_path}. "
            f"Please download the Fashion MNIST CSV files and place them in the '{data_folder}' directory."
        )

    df = pd.read_csv(file_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    if normalize:
        X = X.astype(np.float32) / 255.0
    
    return X, y
