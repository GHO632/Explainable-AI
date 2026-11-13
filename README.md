# Explainability of Black-Box ML Models (XAI)

This repository demonstrates and compares six widely used **Explainable AI (XAI)** methods applied to a Feedforward Neural Network trained on the **UCI Adult Income** dataset. It accompanies a presentation introducing the motivation, taxonomy, mathematical intuition, trade-offs, and practical implementation of modern post-hoc explainability techniques.

---

## Repository Contents

```text
.
├── explainability_adult.ipynb   # Jupyter Notebook: MLP training + 6 XAI methods
├── adult.data                   # UCI Adult dataset (required to run the notebook)
├── lime_explanation.png         # Generated LIME local explanation chart
└── README.md
```

---

## Use Case & Model

- **Dataset:** [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult) (`adult.data`) — binary classification task predicting whether an individual earns `>50K` or `<=50K` per year based on 14 demographic and employment features (`Age`, `Workclass`, `fnlwgt`, `Education`, `Education-Num`, `Marital Status`, `Occupation`, `Relationship`, `Race`, `Sex`, `Capital Gain`, `Capital Loss`, `Hours per week`, `Country`).
- **Preprocessing:**
  - Label encoding for the 8 categorical features (`Workclass`, `Education`, `Marital Status`, `Occupation`, `Relationship`, `Race`, `Sex`, `Country`).
  - Stratified 80/20 train-test split (`random_state=42`).
  - Feature standardization via `StandardScaler` (`X_train_s`, `X_test_s`) for neural network training.
- **Black-Box Model:** Keras `Sequential` Feedforward Neural Network:
  - `Input(14)` → `Dense(64, relu)` → `Dense(32, relu)` → `Dense(1, sigmoid)`
  - Optimizer: `adam` | Loss: `binary_crossentropy` | `epochs=15`, `batch_size=256`, `validation_split=0.1`

---

## Implemented XAI Methods

The notebook implements and visualizes **six explainability techniques** spanning **four methodological families** (with local explanations illustrated primarily on test instance `0`, $p(y=1 \mid x) = 0.237$):

| Category | Method | Implementation | Key Idea |
| :--- | :--- | :--- | :--- |
| **Local Surrogate** | **LIME** | `lime.lime_tabular.LimeTabularExplainer` | Fits a sparse local linear model on perturbed samples weighted by proximity to the instance. |
| **Local Surrogate** | **Anchors** | `anchor.anchor_tabular.AnchorTabularExplainer` | Finds high-precision `IF-THEN` rules (e.g., `Relationship = 3 AND Marital Status = 4`, Precision `0.98`, Coverage `0.13`). |
| **Occlusion / Perturbation** | **SHAP** | `shap.KernelExplainer` | Approximates game-theoretic Shapley values across feature coalitions; provides global **Beeswarm** and local **Waterfall** plots. |
| **Occlusion / Perturbation** | **PDA** *(Prediction Difference Analysis)* | Custom NumPy / Keras implementation | Marginalizes each feature individually using `n_samples=200` draws from training data and measures $p_0 - \mathbb{E}[p(y=1 \mid x_{\setminus j})]$. |
| **Gradient-Based** | **Integrated Gradients (IG)** | Custom TensorFlow `GradientTape` implementation | Integrates gradients along a linear path (`m_steps=50`) from the training-mean baseline to input $x$. |
| **Propagation-Based** | **LRP** *(Layer-wise Relevance Propagation, $\varepsilon$-rule)* | Custom NumPy backward pass (`eps=1e-6`) | Conservatively redistributes the predicted output probability $p(y=1 \mid x_0)$ layer by layer back to the input features. |

---

## Getting Started

### 1. Requirements

- Python 3.9+
- Jupyter Notebook / JupyterLab
- Required packages:
  ```bash
  pip install numpy pandas scikit-learn tensorflow matplotlib lime anchor-exp shap
  ```

### 2. Dataset Setup

Download `adult.data` from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult) and place it in the same directory as the `.ipynb` notebook.

### 3. Run the Notebook

Launch Jupyter and execute the cells sequentially:

```bash
jupyter notebook explainability_adult.ipynb
```

---

## Presentation Overview

The accompanying slide deck covers:
1. **Motivation & Context:** Why black-box models need explainability in high-stakes domains (trust & adoption, validation & bias detection, regulatory governance, model improvement).
2. **Method Taxonomy:** Local surrogates, occlusion/perturbation, gradient-based, and propagation-based attribution.
3. **Deep Dives on Each Method:** Intuition, mathematical formulation, strengths/limitations, code snippets, and reference repositories:
   - LIME: [`marcotcr/lime`](https://github.com/marcotcr/lime)
   - Anchors: [`marcotcr/anchor`](https://github.com/marcotcr/anchor)
   - SHAP: [`shap/shap`](https://github.com/shap/shap)
   - PDA: [`lmzintgraf/DeepVis-PredDiff`](https://github.com/lmzintgraf/DeepVis-PredDiff)
   - Integrated Gradients: [`ankurtaly/Integrated-Gradients`](https://github.com/ankurtaly/Integrated-Gradients)
   - LRP: [`chr5tphr/zennit`](https://github.com/chr5tphr/zennit)
