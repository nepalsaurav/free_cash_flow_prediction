# Academic References and Literature

This document compiles the foundational academic literature used to design the machine learning architecture, feature engineering, and evaluation methodology for this study.

### 1. Target Variable Engineering (Growth vs Absolute) and Model Selection
*   **Kampouridis, E., et al.** "Application of Machine Learning Algorithms to Free Cash Flows Growth Rate Estimation."
    *   *Relevance:* This is the primary benchmark paper justifying the shift from predicting absolute cash flows to predicting the Free Cash Flow **Growth Rate**. It explicitly addresses the "small sample problem" in fundamental corporate datasets (under 1,000 rows). 
    *   *Contrast:* While this paper found K-Nearest Neighbors (KNN) to be highly effective, our empirical testing on the Nepalese Hydropower sector directly challenged this, proving XGBoost vastly outperformed KNN for physics-driven infrastructure assets.

### 2. The Dominance of Tree-Based Ensembles in Tabular Finance
*   **General Literature on XGBoost and LightGBM in Corporate Finance**
    *   *Relevance:* Academic consensus dictates that for small-to-medium sized tabular financial data, Gradient Boosting frameworks (XGBoost, LightGBM) drastically outperform Deep Learning (LSTMs, Neural Networks). The literature highlights XGBoost`s level-wise tree growth as highly resistant to overfitting on noisy financial data compared to Random Forest.

### 3. Explainable AI (XAI) in Financial Modeling
*   **Lundberg, S. M., & Lee, S. I.** "A Unified Approach to Interpreting Model Predictions" (SHAP).
    *   *Relevance:* Modern financial journals heavily penalize "black-box" machine learning models. The integration of SHAP (SHapley Additive exPlanations) summary plots in our methodology satisfies the academic requirement for XAI. It proves *why* the XGBoost model made specific FCF forecasts based on underlying operational metrics (PLF) and debt structures, allowing for interpretable Scenario Analysis.

### 4. Feature Selection vs. Feature Extraction
*   **Literature on handling Multicollinearity in Decision Trees**
    *   *Relevance:* Justifies our decision to actively avoid Principal Component Analysis (PCA). PCA destroys the original feature names, rendering SHAP plots unreadable to human reviewers. We utilized `SelectFromModel` to mathematically prune statistical noise (Inflation, Revenue) while preserving full explainability.
