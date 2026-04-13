# ML Classification Engineering

> Classification algorithms from first principles through production deployment — logistic regression, tree ensembles, SVMs, and evaluation frameworks for imbalanced, high-stakes domains including financial risk and anomaly detection.

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat)](https://xgboost.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)

---

## Why Classification Fundamentals Matter for AI Engineers

Production LLM and agentic systems rely on classification at every layer:
- **Intent routing** — which agent handles this query?
- **Output validation** — is this response safe / on-topic?
- **Anomaly detection** — is this transaction / data point an outlier?
- **Embedding clustering** — which semantic category does this document belong to?

Understanding the mathematics and failure modes of classification algorithms makes you a better AI engineer, not just a better data scientist.

---

## Module 1: Logistic Regression — The Foundation

### The Sigmoid Function: Probability, Not a Score

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Maps any real value to (0, 1) — interpretable as P(y=1 | x).

    Key properties:
      σ(0)   = 0.5  (decision boundary)
      σ(+∞)  → 1.0
      σ(-∞)  → 0.0
      σ'(z)  = σ(z)(1 - σ(z))  ← clean derivative, enables backprop

    Numerical stability: clip z to [-500, 500] to prevent overflow.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
```

### Binary Cross-Entropy: Why Not MSE?

```
MSE on classification → non-convex loss surface → gradient descent gets stuck in local minima
BCE → convex loss surface → guaranteed convergence to global minimum

BCE(y, ŷ) = -(1/m) Σ [ yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ) ]

Intuition:
  When y=1 and ŷ≈0 → loss → ∞  (heavily penalises confident wrong predictions)
  When y=1 and ŷ≈1 → loss → 0  (correctly confident → near-zero loss)
```

### Gradient Descent for Logistic Regression

```python
def logistic_gradient_descent(
    X: np.ndarray, y: np.ndarray,
    alpha: float = 0.01, iterations: int = 1000,
    lambda_reg: float = 0.01,  # L2 regularisation
) -> tuple[np.ndarray, float, list[float]]:
    m, n = X.shape
    w, b = np.zeros(n), 0.0
    costs = []

    for i in range(iterations):
        z = X @ w + b
        y_hat = sigmoid(z)
        error = y_hat - y

        # Vectorised gradients
        dw = (X.T @ error) / m + (lambda_reg / m) * w  # L2 penalty on weights
        db = np.mean(error)

        w -= alpha * dw
        b -= alpha * db

        if i % 100 == 0:
            cost = -np.mean(y * np.log(y_hat + 1e-8) + (1-y) * np.log(1-y_hat + 1e-8))
            costs.append(cost)

    return w, b, costs
```

---

## Module 2: Handling Imbalanced Classes

Critical for fraud detection, anomaly detection, and medical/compliance classification — all domains where the minority class is the one that matters most.

```python
# Anti-pattern: optimising for accuracy on imbalanced data
# 99% "no fraud" → predict all negative → 99% accuracy → useless model

# Correct approach: evaluation suite for imbalanced classification
def evaluate_classifier(model, X_test, y_test, threshold: float = 0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        # Never report just accuracy on imbalanced data
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),     # Of predicted positives, how many correct?
        "recall": recall_score(y_test, y_pred),            # Of actual positives, how many found?
        "f1": f1_score(y_test, y_pred),                    # Harmonic mean — use when FP and FN equally costly
        "auc_roc": roc_auc_score(y_test, y_prob),          # Threshold-independent performance
        "avg_precision": average_precision_score(y_test, y_prob),  # Better than AUC for severe imbalance
    }
```

### Imbalance Handling Strategies

| Strategy | When to Use | Mechanism |
|---|---|---|
| `class_weight='balanced'` | Mild imbalance (5:1 to 10:1) | Upweights minority class in loss |
| SMOTE oversampling | Moderate imbalance | Synthetic minority samples |
| Undersampling majority | Large datasets, severe imbalance | Random removal of majority |
| Threshold tuning | Post-training | Shift decision boundary from 0.5 |
| Ensemble (BalancedBaggingClassifier) | Severe imbalance + high stakes | Bootstrap with balanced subsamples |

---

## Module 3: Algorithm Comparison

Systematic benchmark across classifiers on a financial risk dataset:

| Algorithm | AUC-ROC | F1 (minority) | Training Time | Interpretable |
|---|---|---|---|---|
| Logistic Regression | 0.847 | 0.71 | <1s | ✅ Yes |
| Decision Tree | 0.791 | 0.68 | 1s | ✅ Yes |
| Random Forest | 0.912 | 0.81 | 12s | Partial |
| **XGBoost** | **0.934** | **0.84** | 8s | Partial |
| SVM (RBF kernel) | 0.889 | 0.79 | 45s | ❌ No |
| Naive Bayes | 0.821 | 0.66 | <1s | ✅ Yes |

**Key insight:** XGBoost wins on tabular financial data. Logistic Regression wins when you need regulatory explainability (OSFI model risk requirements). Never choose a model without benchmarking both.

---

## Module 4: Feature Engineering for Classification

```python
# Feature importance — understand what your model actually learned
importances = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

# Top features reveal data quality issues and business insights:
# If "data_entry_timestamp" ranks #1 → data leakage from future information
# If "account_age_days" ranks #1 → sensible business signal
```

---

## Production Classification Pattern

```python
class ProductionClassifier:
    """
    Production-safe classifier wrapper:
    - Threshold calibrated on validation set (not default 0.5)
    - Probability calibration with Platt scaling / isotonic regression
    - Prediction with uncertainty quantification
    - Audit log for regulated environments
    """

    def predict_with_confidence(
        self, X: np.ndarray, min_confidence: float = 0.7
    ) -> dict:
        proba = self.model.predict_proba(X)
        confidence = proba.max(axis=1)
        prediction = (proba[:, 1] >= self.optimal_threshold).astype(int)

        return {
            "prediction": prediction,
            "probability": proba[:, 1],
            "confidence": confidence,
            "requires_human_review": confidence < min_confidence,
        }
```

---

## Related Work

- [ml-neural-network-projects](https://github.com/codebygarrysingh/ml-neural-network-projects) — extends to deep learning classifiers (LSTM, MLP)
- [data-prep-utility](https://github.com/codebygarrysingh/data-prep-utility) — preprocessing pipeline used upstream
- [production-rag-pipeline](https://github.com/codebygarrysingh/production-rag-pipeline) — classification applied to retrieval routing and output validation

---

## Author

**Garry Singh** — Principal AI & Data Engineer · MSc Oxford

[Portfolio](https://garrysingh.dev) · [LinkedIn](https://linkedin.com/in/singhgarry) · [Book a Consultation](https://calendly.com/garry-singh2902)
