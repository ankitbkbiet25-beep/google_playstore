# Google Play Store Rating Predictor: Strategic Classification Ensemble

## Project Overview

This project implements a high-performance system to classify Google Play Store apps into **High-Rating** or **Low-Rating** groups. It showcases an essential data science skill: **iterative problem-solving and strategic reframing.**

The final model provides actionable intelligence for app developers and investors to gauge market quality and potential.

## The Strategic Pivot (From Failure to Success)

The most critical part of this project was the decision to pivot the problem, as documented across multiple notebooks:

### Phase 1: Initial Regression Attempt (Failure)
* **Notebook:** `googleplaystoreEDA.ipynb`
* **Goal:** Predict the continuous `Rating` value (e.g., 4.2, 4.5) using Regression models (Linear Regression, Random Forest Regressor).
* **Result:** **Very Low $R^2$ Score ($\sim 0.11$)** across all models, proving the approach was ineffective due to noisy and unbalanced data.

### Phase 2: Reframing to Classification (Success)
* **Notebook:** `allin.ipynb`
* **Goal:** Convert the problem into a binary classification task: **High Rating** ($\text{Rating} > 4.0$) vs. **Low Rating** ($\text{Rating} \le 4.0$).
* **Action:** Applied **SMOTETomek** to address severe class imbalance, stabilizing the training process.
* **Result:** Established a viable baseline with robust performance using various base classifiers (XGBoost, CatBoost, etc.).

---

## Final Model & Technology Stack

### Final Ensemble Model
The final solution utilized a **Soft Voting Classifier** (Ensemble) combining three powerful, individually tuned models to ensure the highest stability and generalizability:
1.  **XGBoost Classifier**
2.  **CatBoost Classifier**
3.  **Random Forest Classifier**

### Technology
| Component | File / Tool | Function |
| :--- | :--- | :--- |
| **Preprocessing** | `BasicPreprocessing.py` | Custom functions for cleaning and converting messy string fields (`Size`, `Installs`, `Price`) into usable numeric data. |
| **Final Workflow** | `model.ipynb` | Contains the **Pipeline creation**, **Ensemble construction**, and **SHAP** analysis for model interpretability. |
| **Deployment** | `main.py` | Interactive **Streamlit dashboard** for real-time prediction and visualization, summarizing the project's journey. |

---

## Final Performance Metrics

The ensemble model achieved strong results on the high-stakes classification task:

| Metric | Value | Insight |
| :--- | :--- | :--- |
| **Accuracy** | $\mathbf{\sim 72.0\%}$ | Overall correctness of the model. |
| **F1-Score (Weighted)** | $\mathbf{\sim 81.2\%}$ | Excellent balance between Precision and Recall for the High/Low Rating classes. |
| **Reference:** `model_metrics_table.csv`

## Repository Structure & Execution

The repository is structured to be immediately runnable.

```

.
├── main.py                          \<-- Primary Streamlit Dashboard Executable
├── BasicPreprocessing.py            \<-- Custom Preprocessing Module
├── googleplaystore.csv              \<-- App Dataset
├── model.ipynb                      \<-- Final Pipeline & SHAP Analysis
├── allin.ipynb                      \<-- Classification Baseline Testing (Phase 2)
├── googleplaystoreEDA.ipynb         \<-- Initial EDA & Regression Failure (Phase 1)
├── model\_metrics\_table.csv          \<-- Final Performance Report
└── voting\_classifier\_pipeline.pkl   \<-- Saved Ensemble Model Pipeline (Generated asset)

````

### How to Run the Dashboard
1.  **Install Dependencies:** Ensure all required libraries (streamlit, joblib, sklearn, xgboost, catboost, etc.) are installed.
2.  **Run Application:** Execute the main Streamlit file from your terminal:
    ```bash
    streamlit run main.py
    ```