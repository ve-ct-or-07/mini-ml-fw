import numpy as np
import os

# --- Boilerplate: Imports ---
# Import necessary modules from your library and standard libraries
try:
    from my_ml_lib.datasets import load_spambase, DatasetNotFoundError
    from my_ml_lib.preprocessing import StandardScaler
    from my_ml_lib.linear_models.classification import LogisticRegression
    from my_ml_lib.model_selection import KFold, train_test_split
except ImportError as e:
    print(f"Error importing library components: {e}")
    print("Please ensure your my_ml_lib structure and __init__.py files are correct.")
    exit()
# --- End Boilerplate ---

# --- Boilerplate: Configuration ---
# Define constants for the experiment
DATA_FOLDER = "data/" # Directory containing spambase.data
TEST_SIZE = 0.2       # Proportion of data to use for the final test set
RANDOM_STATE = 42     # Seed for random operations (train/test split, KFold shuffle)
N_SPLITS_CV = 5       # Number of folds for cross-validation
ALPHAS_TO_TEST = [0.01, 0.1, 1, 10, 100] # L2 regularization strengths to test
# --- End Boilerplate ---


# --- Step 1 - Load Data ---
print("Loading Spambase dataset...")
try:
    X, y = load_spambase(DATA_FOLDER)
    print(f"Loaded dataset with shape X: {X.shape}, y: {y.shape}")
except DatasetNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Unexpected error while loading dataset: {e}")
    exit()


# --- Step 2 - Split Data into Train and Test ---
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
)
print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")


# --- Helper Function for Cross-Validation ---
def find_best_alpha(X_train_cv, y_train_cv, alphas, n_splits, random_state):
    """Performs K-Fold CV to find the best alpha for Logistic Regression."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_alpha_found = None
    best_val_score = -np.inf

    for alpha in alphas:
        val_scores = []
        for train_idx, val_idx in kf.split(X_train_cv):
            X_tr, X_val = X_train_cv[train_idx], X_train_cv[val_idx]
            y_tr, y_val = y_train_cv[train_idx], y_train_cv[val_idx]

            model = LogisticRegression(alpha=alpha, lr=0.01)
            model.fit(X_tr, y_tr)
            val_acc = model.score(X_val, y_val)
            val_scores.append(val_acc)

        mean_val_acc = np.mean(val_scores)
        # print(f"Alpha={alpha}, Mean CV Accuracy={mean_val_acc:.4f}")
        if mean_val_acc > best_val_score:
            best_val_score = mean_val_acc
            best_alpha_found = alpha

    return best_alpha_found


# --- Step 4 - Experiment with RAW Data ---
print("\n--- Experiment: Raw Data ---")
best_alpha_raw = find_best_alpha(X_train, y_train, ALPHAS_TO_TEST, N_SPLITS_CV, RANDOM_STATE)
print(f"Best alpha (raw): {best_alpha_raw}")

# --- Step 5 - Train and Evaluate Final RAW Model ---
raw_model = LogisticRegression(alpha=best_alpha_raw, lr=0.01)
raw_model.fit(X_train, y_train)
train_error_raw = 1 - raw_model.score(X_train, y_train)
test_error_raw = 1 - raw_model.score(X_test, y_test)


# --- Step 6 - Experiment with STANDARDIZED Data ---
print("\n--- Experiment: Standardized Data ---")

# Step 6a - Initialize and Fit Scaler
scaler = StandardScaler()
scaler.fit(X_train)
print("StandardScaler fitted on training data.")

# Step 6b - Transform train and test data
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
print("Train and test data transformed using StandardScaler.")

# Step 6c - Find Best Alpha for Standardized Data
best_alpha_std = find_best_alpha(X_train_std, y_train, ALPHAS_TO_TEST, N_SPLITS_CV, RANDOM_STATE)
print(f"Best alpha (standardized): {best_alpha_std}")

# Step 7 - Train and Evaluate Final STANDARDIZED Model
std_model = LogisticRegression(alpha=best_alpha_std, lr=0.01)
std_model.fit(X_train_std, y_train)
train_error_std = 1 - std_model.score(X_train_std, y_train)
test_error_std = 1 - std_model.score(X_test_std, y_test)


# --- Boilerplate: Report Results ---
print("\n--- Summary Results ---")
print(f"Preprocessing | Best Alpha | Train Error | Test Error")
print(f":------------|-----------:|------------:|-----------:")
print(f"Raw           | {best_alpha_raw:<10} | {train_error_raw:<11.4f} | {test_error_raw:<10.4f}")
print(f"Standardized  | {best_alpha_std:<10} | {train_error_std:<11.4f} | {test_error_std:<10.4f}")
print("\nNOTE: Ensure the results above reflect your actual computed values.")
