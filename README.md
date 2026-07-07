# Hydropower Machine Learning & Intrinsic Valuation Pipeline

This repository contains the complete dataset, Machine Learning forecasting pipeline, and empirical Discounted Cash Flow (DCF) valuation models used to forecast Free Cash Flow (FCF) for 105 Nepalese hydropower companies over a 16-year panel (2010–2025).

## 1. Data Provenance and Pipeline
To guarantee full reproducibility, the 14-feature fundamental dataset was manually assembled and rigorously verified. No proprietary "black-box" data terminals were used.
*   **Corporate Financials:** Extracted manually from audited quarterly filings submitted to the Securities Board of Nepal (SEBON) and the Nepal Stock Exchange (NEPSE).
*   **Macroeconomic Data:** Sourced from the Nepal Rastra Bank (NRB) statistical bulletins. Due to systemic data unavailability prior to 2012, early macroeconomic indicators (such as interbank lending proxies for interest rates) were meticulously transcribed from raw historical PDFs.
*   **Systemic Energy Data (NEA):** Extracted from Nepal Electricity Authority (NEA) Annual Reports. 
*   **Log-Linear Backcasting:** Because systemic power variables (National Demand, IPP Capacity) follow exponential compound-growth trajectories in emerging economies, missing historical records (2010-2015) were mathematically imputed using **Log-Linear Ordinary Least Squares (OLS) Backcasting** ($ln(y) = \beta_0 + \beta_1(Year)$) rather than simple linear interpolation.

## 2. Feature Engineering & Target Processing
*   **Target Variable:** The target is the 1-year forward **Percentage Growth Rate** of Free Cash Flow. 
*   **Winsorization:** Because fundamental corporate data can drop near zero, percentage growth calculations can generate massive mathematical spikes. To prevent the Mean Absolute Error (MAE) from exploding, the target was Winsorized at the **5th and 95th percentiles**.
*   **Implied PLF (iPLF):** Due to the lack of digitized operational efficiency data, this study utilizes an Implied Plant Load Factor metric. Since hydroelectric revenue ($R$) is a deterministic function of Installed Capacity ($C$) and Blended PPA Tariff ($T$), we algebraically reverse-engineered operational efficiency: $iPLF = R / (C \times 8760 \times T)$. 
    * *Note on Circularity:* Because iPLF is derived from Revenue and Capacity, robustness checks (training the SVR model without iPLF or Revenue) are included in the Jupyter Notebook to confirm feature stability.

## 3. Machine Learning Methodology
This repository executes a "6-Model Showdown" against a Traditional Naive Baseline (0% growth assumption).
*   **Algorithm:** The Support Vector Regression (SVR) model utilizing a Radial Basis Function (RBF) Kernel achieved the lowest out-of-sample forecasting error.
*   **Cross-Validation:** To prevent data leakage across the panel dataset, all model tuning (`RandomizedSearchCV`) and Learning Curves utilize **GroupKFold** cross-validation, strictly grouped by firm ID (`Ticker`).
*   **Explainable AI (XAI):** Permutation Importance is reported using `n_repeats=30` with standard deviations (error bars) to ensure robustness, accompanied by Partial Dependence Plots (PDP) to map the economic logic of the SVR Kernel.

## 4. Execution
The entire end-to-end pipeline (Data Loading, Exploratory Data Analysis, Model Training, Statistical Testing, and DCF Valuation) is contained within a single executable Jupyter Notebook:
`SVR_Valuation_Pipeline.ipynb`
