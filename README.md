# Foundations of Machine Learning – Assignment 3  
**Mini-ML Library and Autograd Engine (SGD-based Logistic Regression)**  

---

## 📂 Repository Structure
```
my_ml_lib/
│
├── datasets/
│   ├── __init__.py
│   ├── _loaders.py          # load_spambase(), load_fashion_mnist()
│
├── preprocessing/
│   ├── __init__.py
│   ├── _data.py             # StandardScaler
│   ├── _polynomial.py       # PolynomialFeatures
│   ├── _gaussian.py         # GaussianBasisFeatures
│
├── linear_models/
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── _logistic.py     # Logistic Regression (SGD-based)
│
├── model_selection/
│   ├── __init__.py
│   ├── _split.py            # train_test_split, train_test_val_split
│   ├── _kfold.py            # KFold cross-validation
│
├── nn/
│   ├── autograd.py          # Value class with autograd engine
│   ├── modules/
│   │   ├── base.py          # Base Module class
│   │   ├── linear.py        # Linear layer
│   │   ├── activations.py   # ReLU, Sigmoid
│   │   ├── containers.py    # Sequential
│   │── optim.py             # SGD optimizer
│   │── losses.py            # CrossEntropyLoss, BCELoss
│
│── run_spam_experiment.py       # Problem 3 experiment
│── visualize.py                 # Problem 4 computation graph visualization
│
├── capstone_showdown.ipynb      # Problem 5 - Final experiments
├── saved_models/                # Saved model weights (.npz or .pkl)
├── report.pdf                   # Final PDF report
└── README.md
```

---

## ⚙️ Installation & Requirements

```bash
pip install numpy matplotlib graphviz
```

*(Optional)* To view computation graphs, install Graphviz system package:  
- **Ubuntu:** `sudo apt install graphviz`  
- **Windows:** Download from [graphviz.org](https://graphviz.org/download/) and add to PATH.

---

## 🧠 Problem 3 – Spam Classification (SGD Logistic Regression)

### Run:
```bash
python run_spam_experiment.py
```

This:
- Loads the **Spambase** dataset (from `data/spambase.data`)
- Trains logistic regression using **SGD optimization**
- Performs **5-fold cross-validation** to find best L2 regularization strength (α)
- Compares results **with vs. without StandardScaler**

### Output Example:
```
--- Summary Results ---
Preprocessing | Best Alpha | Train Error | Test Error
:------------|-----------:|------------:|-----------:
Raw           | 0.1        | 0.4939      | 0.4859
Standardized  | 0.01       | 0.0826      | 0.0902
```

---

## 🔍 Problem 4 – Autograd Engine Visualization

### Run:
```bash
python visualize.py
```

This generates a computational graph for a tiny MLP (e.g., `Sequential(Linear(2,3), ReLU())`).  
Output is saved as:

```
example_computation_graph.svg
---

## 🧩 Problem 5 – Capstone Showdown (Fashion-MNIST)

Run your experiments and tuning using:

### Jupyter Notebook
```bash
jupyter notebook capstone_showdown.ipynb
```

Your notebook/scripts should:
- Load **Fashion-MNIST**
- Train and tune 5 models:
  1. OvR Logistic Regression (SGD)
  2. Softmax Regression (raw)
  3. Softmax + Polynomial Features
  4. Softmax + Gaussian Basis
  5. MLP (using autograd engine)
- Log **training loss curves** and **final accuracy table**
- Saved your best model in `saved_models/`

---

## 💾 Saved Models

- `saved_models/ovr_logistic.pkl` → Pickled OvR logistic model  
- `saved_models/best_autograd_model.npz` → Saved autograd model weights

You can reload autograd models with:
```python
model2.load_state_dict("saved_models/best_autograd_model.npz")
```

---

## 🧾 Report & Deliverables

Your submission ZIP should include:

| File | Purpose |
|------|----------|
| `report.pdf` | All derivations, plots, tables, and analyses (Q1–Q5) |
| `README.md` | This file (execution instructions) |
| `run_spam_experiment.py` | For Q3 |
| `visualize.py` | For Q4 |
| `capstone_showdown.ipynb` | For Q5 experiments |
| `saved_models/` | Trained best model(s) |
| `my_ml_lib/` | Your implemented library |

---

## 👩‍💻 Notes

- Logistic Regression implemented via **Stochastic Gradient Descent (SGD)** instead of IRLS for scalability.
- Ensure **random seeds** are set for reproducibility (`np.random.seed(42)`).
- Always fit scalers and transformations **only on training data**.

---

**Author:** *P V Charan Teja*  
**Roll Number:** AI24BTECH11022
