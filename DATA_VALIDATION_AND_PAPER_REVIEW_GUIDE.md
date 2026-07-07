# Complete Guide to Data Validation and Paper Review

This guide provides a rigorous, step-by-step checklist to verify every component of your research before submitting it to the **Apex Journal of Business and Management (AJBM)**. Because your paper utilizes advanced Machine Learning (ML) and empirical valuation, reviewers will scrutinize your data integrity, methodology, and the consistency between your code and manuscript.

Follow this guide to guarantee that your paper is computationally reproducible, statistically sound, and flawlessly consistent.

---

## Phase 1: Data Integrity & Pre-processing Verification

Your data is the foundation of the ML models. You must verify that the transformation from raw financial reports to the `master_ml_dataset.csv` is mathematically sound and free of leakage.

### 1. Raw Data Tracing
- [ ] **Sample Size Check:** Confirm that the dataset contains exactly 105 unique Nepalese hydropower companies.
- [ ] **Temporal Check:** Confirm the panel spans from 2010 to 2025.
- [ ] **Source Verification:** Spot-check 3 random companies. Ensure their raw Revenue, Debt, and Working Capital in the CSV exactly match their published Annual Reports.

### 2. Missing Data Imputation (Log-Linear Backcasting)
- [ ] **Method Verification:** Ensure the NEA systemic variables missing between 2010-2015 were mathematically imputed using Log-Linear OLS ($ln(y) = \beta_0 + \beta_1(Year)$), not simple average interpolation. 

### 3. Feature Engineering & Target Variable
- [ ] **FCF Growth Formula:** Verify the target variable calculation in your Python code matches the paper's formula: `(FCF_t+1 - FCF_t) / (|FCF_t| + 1)`.
- [ ] **Winsorization:** Check that the target variable is explicitly winsorized at the 5th and 95th percentiles to prevent infinity/explosion errors.
- [ ] **The 13 Features:** Verify the final pipeline uses exactly 13 features: `Revenue`, `Debt_Ratio`, `ROA`, `Working_Capital`, `CapEx`, `Installed_Capacity_MW`, `License_Term_Years`, `GDP_Growth`, `Inflation`, `Interest_Rate`, `Years_Since_COD`, `Pre_Post_COD`, and `EBIT_Margin`.
- [ ] **Circular Logic Check (CRITICAL):** Open your final pipeline script (`SVR_Valuation_Pipeline.ipynb` or `final_pipeline.py`) and do a strict search (Ctrl+F) for `iPLF` (Implied Plant Load Factor). Ensure it is **100% excluded** from the `X` (features) matrix.

---

## Phase 2: ML Methodology & Code Verification

Reviewers will check if your ML implementation "cheated" by leaking data or overfitting.

### 1. Data Leakage Prevention (GroupKFold)
- [ ] **Group Splitting:** In your pipeline, verify that cross-validation uses `GroupKFold(n_splits=3)` (or 5) and that the `groups` parameter is strictly mapped to the **Firm ID/Company Name**. Standard `KFold` is an automatic rejection for panel data because it leaks future data of the same company into the training set.

### 2. Scaling
- [ ] **Pipeline Order:** Ensure `StandardScaler` is applied *inside* the cross-validation loop (or pipeline), preventing data leakage from the test set into the training set's mean/variance calculations.

### 3. Hyperparameter Tuning
- [ ] **SVR Setup:** Confirm that the Support Vector Regression uses the Radial Basis Function (`kernel='rbf'`).
- [ ] **Search Space:** Verify `RandomizedSearchCV` was used to tune `C`, `gamma`, and `epsilon`.

---

## Phase 3: Explainable AI (XAI) Verification

Your interpretation of the models must exactly match the physics of the generated plots.

### 1. Permutation Importance
- [ ] **Repeats:** Verify that permutation importance was run with `n_repeats=30` (or similar high number) to generate the error bars ($\pm$ SD).
- [ ] **Rankings:** Check `plots/feature_importance.png`. Does `ROA` rank #1? Does `Pre_Post_COD` rank in the top 4? Does `GDP_Growth` rank near the bottom? Ensure the text matches the visual rankings.

### 2. Partial Dependence Plots (PDP)
- [ ] **ROA Plot Check:** Look at the PDP for `ROA`. It should show a *downward* slope. Verify that the paper correctly interprets this as "plant maturity" (older plants have high ROA but low percentage growth).
- [ ] **Pre_Post_COD Plot Check:** The PDP for `Pre_Post_COD` should show a jump/positive shift from 0 (Pre) to 1 (Post).

---

## Phase 4: Valuation Mechanics Verification

The DCF math must be financially sound to pass business/finance reviewers.

### 1. The DCF Math
- [ ] **WACC:** Confirm the Discount Rate applied is 10%.
- [ ] **Terminal Value (Finite-Horizon):** Check the valuation script. Ensure the Terminal Value does *not* assume perpetual growth, but is strictly bounded by `License_Term_Years`.
- [ ] **Enterprise to Equity:** Verify that the calculation takes the Present Value of the FCFF (Enterprise Value) and **subtracts Total Debt** to arrive at the Intrinsic Equity Value.

### 2. Empirical Market Baseline
- [ ] **NEPSE Snapshot:** Confirm the actual market capitalizations used for the benchmark exactly match NEPSE data for Q4 2023.

---

## Phase 5: Manuscript Consistency (The Final Polish)

The #1 reason for "Major Revision" verdicts is when a number in a table doesn't match a number in the text.

### 1. Numerical Consistency (Checklist)
Cross-reference these exact numbers across the Abstract, Table 1, and the body text. They must match perfectly:
- [ ] **SVR MAE:** `Rs 91,442`
- [ ] **Wilcoxon p-value:** `p = 0.00008` (or `p < 0.001`)
- [ ] **Paired t-test p-value:** `p = 0.062`
- [ ] **DCF Improvement:** `3.95%`
- [ ] **Traditional DCF Avg Error:** `Rs 5.73 Million` (or `5,736,624`)
- [ ] **SVR DCF Avg Error:** `Rs 5.51 Million` (or `5,510,246`)

### 2. Figure and Table Alignment
- [ ] **Table 1 vs Figure 2:** Verify that Table 1 lists 6 models (Traditional Baseline, SVR, XGBoost, LightGBM, KNN, Random Forest, Ridge) and that `plots/model_comparison.png` shows exact matching bars for all of them.
- [ ] **Figure Captions:** Ensure every Figure (1 through 7) has a bolded caption below it that matches the text references.

### 3. Formatting
- [ ] **Double Column:** Ensure the submitted MS Word file is formatted as a double-column layout (as generated in your latest `paper.docx`).
- [ ] **Abstract Structure:** Verify the abstract uses the exact bolded sub-headers requested by AJBM (`Purpose:`, `Design/Methodology/Approach:`, etc.).

---
**Final Recommendation before submission:**
Run the `SVR_Valuation_Pipeline.ipynb` from top to bottom one last time. If the script executes without errors and produces the exact numbers listed in Table 1, your paper is 100% verified, computationally reproducible, and ready to submit!
