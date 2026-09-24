# Used Car Valuation & Price Prediction Engine

## Overview

This repository delivers the complete predictive modeling and analytical pipeline for **Required Assignment 11.1: What Drives the Price of a Car?**  
My objective is to identify the underlying macroeconomic and vehicle-specific drivers of used-car prices and build a reliable, regularized regression engine to power automated trade-in appraisals and inventory acquisition strategies.

Following the structured CRISP-DM framework, I processed over $426,000$ raw market listings, engineered polynomial depreciation features, benchmarked 11 model variations (OLS, Ridge, and Lasso), and diagnosed real-world boundary conditions to formulate actionable operational guardrails.

![Machine Learning Workflow](./images/kurt.jpeg)

## Notebook

**Main Analysis & Production Notebook:**  
[`notebooks/price_of_car.ipynb`](./notebooks/price_of_car.ipynb)

What's Included:

- Systematic CRISP-DM workflow execution with clean markdown sectioning
- Data pruning and hierarchical imputation reducing $426,880$ rows to $329,752$ records
- Continuous polynomial degree optimization (Degrees 1 through 5)
- Full-feature benchmarking across baseline OLS, Ridge (\(\alpha \in \{1, 10, 100, 1000\}\)), and 5-fold cross-validated Lasso
- Diagnostic visualizations (100-sample tracking curves and \(0–\$80\text{k}\) core test residual scatter plots)
- Comprehensive feature impact tables, unscaled dollar decay rates, and production guardrail logic

## Repository Structure

```text
.
├── README.md                           # Executive briefing, model benchmarks, and findings
├── notebooks/
│   └── price_of_car.ipynb              # Main analytical and modeling notebook
├── data/
│   ├── vehicles.csv                    # Raw dataset (~426k listings)
│   └── output/
│       ├── dataframes/                 # Checkpointed Parquet datasets across all phases
│       └── models/                     # Serialized joblib pipelines (OLS, Ridge, Lasso)
└── images/
    └── plots/                          # High-resolution EDA and model diagnostic plots
```

## Running the Notebook

1. **Environment Setup:**  
   Open `notebooks/price_of_car.ipynb` in your preferred Python environment (Jupyter Notebook, JupyterLab, VS Code, or Google Colab) with standard data science packages installed (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, and `pyarrow`).

2. **File Path Configuration (Google Colab / Drive vs. Local):**  
   The notebook includes setup cells configured to mount Google Drive for execution in Google Colab. If you are running the notebook locally, comment out the Drive-mounting cells and set the working path to the local repository root so it can read `data/vehicles.csv` and write to `data/output/`.

3. **Execution:**  
   Run all cells sequentially from top to bottom.

