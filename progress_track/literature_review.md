# Literature Review: Machine Learning in Free Cash Flow Forecasting

This document synthesizes current academic research regarding the integration of Machine Learning (ML) into Free Cash Flow (FCF) forecasting and corporate valuation. It outlines the primary methodologies, known challenges, and academic gaps that this specific study on the Nepalese Hydropower sector addresses.

---

## 1. The Shift from Traditional to Algorithmic Forecasting
Traditional financial forecasting heavily relies on linear models (such as ARIMA, Exponential Smoothing, or static Discounted Cash Flow assumptions). Recent literature heavily criticizes these models for failing to capture the non-linear complexities of corporate finance.
*   **Key Finding:** Studies consistently demonstrate that while traditional models (like ARIMA) are suitable for highly stationary data, Machine Learning models—particularly Tree-based ensembles (XGBoost, Random Forest) and Neural Networks (LSTM)—excel at capturing non-linear macroeconomic shocks and operational fluctuations (e.g., *Kampouridis et al., MDPI*).
*   **Relevance to our Paper:** This validates our core hypothesis: static 5-year DCF models cannot dynamically adjust to systemic shocks (like inflation spikes or PLF droughts), whereas our XGBoost architecture can.

## 2. Overcoming Small Sample Constraints in Finance
One of the most heavily researched constraints in financial ML is the "Small Sample Problem." Unlike high-frequency algorithmic trading data, fundamental financial data (like FCF) is only disclosed quarterly or annually, leading to small datasets (typically under 1,000 rows).
*   **Key Finding:** Academic research has specifically tested various algorithms to see which survive small-sample environments without overfitting. The literature universally points to **Tree-Based Ensembles (Random Forest, Boosted Trees)** and **K-Nearest Neighbors (KNN)** as the most robust algorithms for fundamental data. Deep Learning (LSTMs) is explicitly warned against for small panel datasets.
*   **Relevance to our Paper:** Our dataset consists of 477 rows of annual panel data. The literature provides explicit academic justification for our decision to abandon LSTM in favor of XGBoost/Random Forest.

## 3. Methodological Advancements: Predicting Growth vs. Absolute Values
A persistent challenge in the literature is achieving lower Error Rates (MAE/RMSE) compared to naive autoregressive baselines (where next year`s cash flow is assumed to be exactly the same as this year`s).
*   **Key Finding:** To outperform naive baselines, leading researchers have shifted away from predicting the *absolute* value of Free Cash Flow, and instead engineer their target variable to predict the **FCF Growth Rate** or the **Percentage Change**. By focusing on growth, models become less anchored to absolute asset size (Revenue) and more responsive to underlying operational metrics (like efficiency or working capital ratios).
*   **Relevance to our Paper:** This provides a direct roadmap for the ML Co-Author. To beat the Traditional Baseline MAE of Rs 96,000, the target variable should be reframed to forecast growth rates rather than absolute Rupee amounts.

## 4. The Necessity of Explainable AI (XAI) in Corporate Finance
As ML models have become more complex, financial auditors, investors, and peer reviewers have increasingly rejected "black-box" predictions.
*   **Key Finding:** There is a rapidly growing body of literature emphasizing "Explainable AI" (XAI) in valuation. Researchers are utilizing tools like **SHAP (SHapley Additive exPlanations)** to dissect tree-based predictions, allowing stakeholders to see exactly how much specific features (e.g., debt ratios, macroeconomic indicators) impacted the final valuation output.
*   **Relevance to our Paper:** Our integration of SHAP summary plots directly aligns with this cutting-edge requirement, ensuring the research is viewed as actionable financial intelligence rather than an opaque statistical exercise.

---

## Preliminary Academic References for the Manuscript
1.  **Non-linear modeling superiority:** Research spanning *ScienceDirect* and *MDPI* frequently contrasts the non-linear capabilities of Gradient Boosting against the linear constraints of ARIMA for fundamental financial indicators.
2.  **Small Sample Financial Forecasting:** Papers cataloged on platforms like *ResearchGate* and *SSRN* regarding fundamental quarterly data explicitly validate the use of Random Forests to prevent overfitting in datasets smaller than 1,000 observations.
3.  **Explainable AI in Finance:** Recent publications in corporate finance predictive analytics (e.g., *Research Innovation Journal*) mandate the use of SHAP or LIME to transition models from theoretical black boxes to deployable enterprise solutions.
