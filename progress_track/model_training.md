# Machine Learning Model Training Details

## 1. Objective and Target Engineering
The primary objective of the machine learning architecture is to forecast the 1-year forward **Free Cash Flow (FCF)** for Nepalese Hydropower companies. Based on academic literature, the target variable ($Y$) was engineered as the **Percentage Growth Rate** (`Target_FCF_Growth`) rather than the absolute cash flow. This prevents the model from anchoring predictions to sheer company size (Revenue) and forces it to learn from operational efficiency.

### Outlier Handling (Winsorizing & Target Distribution)
Because fundamental financial data can drop near zero, percentage growth calculations can generate massive mathematical spikes (e.g., 10,000% growth). To prevent the MAE from exploding, the `Target_FCF_Growth` was Winsorized (clipped) at the 5th and 95th percentiles before training. 
The extreme non-normal distribution (heavy skew and fat tails) mathematically justified abandoning linear ARIMA models in favor of non-linear algorithms.

## 2. Feature Selection (None)
To preserve Explainable AI (XAI) and avoid the "black-box" nature of algorithmic reduction, **all 14 original macroeconomic and operational features** were retained. The SVR model's RBF kernel was relied upon to naturally separate signal from noise without deleting valuable economic context.

## 3. The 6-Model Showdown (Methodology)
To ensure academic rigor, 6 distinct algorithms were evaluated simultaneously against a Traditional Naive Baseline (which assumes 0% growth). 
Because models like KNN and SVR rely on distance metrics, a `StandardScaler` was applied to the selected features prior to training.
All models underwent aggressive hyperparameter tuning using `RandomizedSearchCV` with 3-fold cross-validation to prevent overfitting on the small dataset.

### Models Evaluated:
1. **Ridge Regression:** Regularized linear model (tuned `alpha`).
2. **K-Nearest Neighbors (KNN):** Distance-based baseline recommended by prior literature.
3. **Support Vector Regression (SVR):** Non-linear kernel baseline.
4. **Random Forest:** Ensemble Bagging architecture.
5. **LightGBM:** Fast, leaf-wise Gradient Boosting architecture.
6. **XGBoost:** Level-wise Gradient Boosting architecture.

## 4. Final Evaluation Leaderboard (Reconstructed MAE)
The models predicted the growth rate, which was then mathematically reconstructed back into Absolute Rupees for a fair comparison against the Traditional Baseline.

*Traditional Baseline MAE: Rs 96,114*

| Rank | Model | Reconstructed MAE (Error) |
| :--- | :--- | :--- |
| **1** | **Support Vector Regression (SVR)** | **Rs 91,773** |
| 2 | XGBoost | Rs 95,144 |
| 3 | LightGBM | Rs 99,202 |
| 4 | Random Forest | Rs 108,302 |
| 5 | K-Nearest Neighbors | Rs 112,729 |
| 6 | Ridge Regression | Rs 128,035 |

## 5. The Small Sample Problem (Learning Curve Validation)
Because the dataset only consists of 477 firm-years, tree-based models like Random Forest and LightGBM overfit to the macroeconomic noise. The SVR algorithm utilizing the "Kernel Trick" (RBF Kernel) was the only architecture capable of separating the operational signal from the macroeconomic noise.
To prove this robustness to reviewers, a **Model Learning Curve** was generated. It mathematically verified that as the sample size increased, the Training Error and Cross-Validation Error cleanly converged, definitively proving the SVR model did not overfit.

## 6. Statistical Significance Testing
In accordance with rigorous academic methodology, the model forecasting errors were subjected to statistical hypothesis testing.

*   **Wilcoxon Signed-Rank Test (p = 0.0001):** Because corporate cash flows severely violate the normality assumptions required by a paired t-test, the robust non-parametric Wilcoxon test was applied. The reduction in forecasting error achieved by the SVR architecture was found to be highly statistically significant at the 1% level, successfully rejecting the null hypothesis.
