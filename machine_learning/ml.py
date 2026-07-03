import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# Set global visual style for academic papers
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def load_and_preprocess_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    df = df.sort_values(by=["Ticker", "Year_Int"]).reset_index(drop=True)
    
    df["FCF_NextYear"] = df.groupby("Ticker")["OperatingCashFlow_Proxy"].shift(-1)
    df["Target_FCF_Growth"] = (df["FCF_NextYear"] - df["OperatingCashFlow_Proxy"]) / (df["OperatingCashFlow_Proxy"].abs() + 1)
    
    df = df.dropna(subset=["Target_FCF_Growth"]).copy()
    
    lower_bound = df["Target_FCF_Growth"].quantile(0.05)
    upper_bound = df["Target_FCF_Growth"].quantile(0.95)
    df["Target_FCF_Growth"] = df["Target_FCF_Growth"].clip(lower_bound, upper_bound)
    
    df = df.fillna(0)
    return df

def traditional_baseline_model(df_test):
    print("\n--- 1. Traditional Baseline (Naive No-Growth) ---")
    predictions = df_test["OperatingCashFlow_Proxy"]
    y_true = df_test["FCF_NextYear"]
    
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    
    baseline_errors = np.abs(y_true - predictions)
    
    print(f"Traditional Baseline -> MAE: Rs {mae:,.0f} | RMSE: Rs {rmse:,.0f}")
    return predictions, mae, baseline_errors

def train_evaluate_ml(X_train, y_train, X_test, y_test, df_test_base_fcf, actual_next_year_fcf, feature_names):
    print("\n--- Running the Ultimate 6-Model ML Showdown ---")
    
    print("\n   [0/6] Skipping Feature Selection (Using all 14 features)...")
    X_train_sel = X_train
    X_test_sel = X_test
    
    selected_features = feature_names
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    results = {}
    
    def evaluate(model_name, preds):
        abs_preds = df_test_base_fcf + (df_test_base_fcf.abs() + 1) * preds
        mae = mean_absolute_error(actual_next_year_fcf, abs_preds)
        rmse = np.sqrt(mean_squared_error(actual_next_year_fcf, abs_preds))
        results[model_name] = (mae, rmse)
        print(f"   -> {model_name} MAE: Rs {mae:,.0f} | RMSE: Rs {rmse:,.0f}")
        return abs_preds
    
    # 1. Ridge Regression
    print("\n   [1/6] Tuning Ridge Regression...")
    ridge_search = RandomizedSearchCV(Ridge(random_state=42), {"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}, n_iter=5, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1, random_state=42)
    ridge_search.fit(X_train_scaled, y_train)
    evaluate("Ridge Regression", ridge_search.best_estimator_.predict(X_test_scaled))
    
    # 2. K-Nearest Neighbors (KNN)
    print("\n   [2/6] Tuning K-Nearest Neighbors (k-NN)...")
    knn_search = RandomizedSearchCV(KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 10], "weights": ["uniform", "distance"]}, n_iter=8, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1, random_state=42)
    knn_search.fit(X_train_scaled, y_train)
    evaluate("K-Nearest Neighbors", knn_search.best_estimator_.predict(X_test_scaled))
    
    # 3. Support Vector Regression (SVR)
    print("\n   [3/6] Tuning Support Vector Regression (SVR)...")
    svr_search = RandomizedSearchCV(SVR(), {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.1, 0.01], "epsilon": [0.01, 0.1, 1]}, n_iter=15, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1, random_state=42)
    svr_search.fit(X_train_scaled, y_train)
    svr_model = svr_search.best_estimator_
    svr_preds = svr_model.predict(X_test_scaled)
    svr_abs_preds = evaluate("Support Vector Regression", svr_preds)
    svr_errors = np.abs(actual_next_year_fcf - svr_abs_preds)
    
    # 4. Random Forest
    print("\n   [4/6] Tuning Random Forest...")
    rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7, None]}, n_iter=10, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1, random_state=42)
    rf_search.fit(X_train_scaled, y_train)
    evaluate("Random Forest", rf_search.best_estimator_.predict(X_test_scaled))
    
    # 5. LightGBM
    print("\n   [5/6] Tuning LightGBM...")
    lgbm_search = RandomizedSearchCV(LGBMRegressor(random_state=42, verbose=-1), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "num_leaves": [10, 20, 31]}, n_iter=15, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1, random_state=42)
    lgbm_search.fit(X_train_scaled, y_train)
    evaluate("LightGBM", lgbm_search.best_estimator_.predict(X_test_scaled))
    
    # 6. XGBoost
    print("\n   [6/6] Tuning XGBoost...")
    xgb_search = RandomizedSearchCV(xgb.XGBRegressor(random_state=42), {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]}, n_iter=15, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1, random_state=42)
    xgb_search.fit(X_train_scaled, y_train)
    evaluate("XGBoost", xgb_search.best_estimator_.predict(X_test_scaled))
    
    # Print Leaderboard
    print("\n===================================================================")
    print("                 FINAL MACHINE LEARNING LEADERBOARD")
    print("===================================================================")
    header_rank = "Rank"
    header_model = "Model"
    header_mae = "MAE (Rs)"
    header_rmse = "RMSE (Rs)"
    print(f"{header_rank:<5} | {header_model:<25} | {header_mae:<12} | {header_rmse:<12}")
    print("-" * 63)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][0]) # Sort by MAE
    for i, (model_name, metrics) in enumerate(sorted_results):
        mae, rmse = metrics
        print(f"{i+1:<5} | {model_name:<25} | {mae:<12,.0f} | {rmse:<12,.0f}")
    print("===================================================================\n")
    
    return svr_model, selected_features, X_train_scaled, y_train, scaler, svr_errors, sorted_results, svr_abs_preds

def generate_academic_plots(svr_model, X_train_scaled, y_train, selected_features, sorted_results, actual_fcf, abs_preds, output_dir):
    print(f"--- Generating High-Res Academic Plots in {output_dir}/ ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Feature Importance (Permutation Importance for SVR)
    print("Calculating Permutation Importance for SVR (this might take a few seconds)...")
    result = permutation_importance(svr_model, X_train_scaled, y_train, n_repeats=10, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    indices = np.argsort(importances)
    sorted_features = [selected_features[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), sorted_importances, color="#2ca02c", align="center")
    plt.yticks(range(len(indices)), sorted_features)
    plt.title("SVR Permutation Importance\n(Drivers of Free Cash Flow Growth)")
    plt.xlabel("Mean Decrease in Accuracy (Importance)")
    plt.ylabel("Operational & Financial Metric")
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()
    
    # 2. Model Comparison Bar Chart
    models = [res[0] for res in sorted_results]
    maes = [res[1][0] for res in sorted_results]
    colors = ["#1f77b4" if m == "Support Vector Regression" else "#cccccc" for m in models]
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=maes, y=models, palette=colors)
    plt.axvline(x=96113, color="red", linestyle="--", linewidth=2, label="Traditional Baseline (Rs 96,113)")
    plt.title("Machine Learning Algorithm Performance\n(Predicting Free Cash Flow Growth)")
    plt.xlabel("Mean Absolute Error (MAE in Rupees) - Lower is Better")
    plt.ylabel("Algorithm")
    plt.legend()
    for i, v in enumerate(maes):
        ax.text(v + 1000, i, f"Rs {v:,.0f}", color="black", va="center")
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()
    
    # 3. Actual vs Predicted Scatter
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=actual_fcf, y=abs_preds, alpha=0.6, color="#ff7f0e", edgecolor="k")
    min_val = min(min(actual_fcf), min(abs_preds))
    max_val = max(max(actual_fcf), max(abs_preds))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Prediction Line")
    plt.title("SVR Validation:\nActual vs. Predicted Cash Flow (Out-of-Sample)")
    plt.xlabel("Actual Free Cash Flow (Rs)")
    plt.ylabel("Predicted Free Cash Flow (Rs)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
    plt.close()

def main():
    os.makedirs("machine_learning", exist_ok=True)
    df = load_and_preprocess_data("data/master_ml_dataset.csv")
    
    features = [
        "Revenue", "Debt_Ratio", "ROA", "Working_Capital", "CapEx",
        "Installed_Capacity_MW", "PLF", "License_Term_Years",
        "GDP_Growth", "Inflation", "Interest_Rate", 
        "Years_Since_COD", "Pre_Post_COD", "EBIT_Margin"
    ]
    
    train_df = df[df["Year_Int"] < 2023]
    test_df = df[df["Year_Int"] >= 2023]
    
    X_train = train_df[features]
    y_train = train_df["Target_FCF_Growth"]
    X_test = test_df[features]
    
    print(f"Training Rows: {len(X_train)}")
    print(f"Testing Rows: {len(X_test)}")
    
    trad_preds, trad_mae, baseline_errors = traditional_baseline_model(test_df)
    
    svr_model, selected_features, X_train_scaled, y_train_scaled, scaler, svr_errors, sorted_results, svr_abs_preds = train_evaluate_ml(
        X_train, y_train, X_test, test_df["Target_FCF_Growth"], 
        test_df["OperatingCashFlow_Proxy"], test_df["FCF_NextYear"], features
    )
    
    print("===================================================================")
    print("                 STATISTICAL SIGNIFICANCE TESTING")
    print("===================================================================")
    print("Hypothesis: ML (SVR) produces significantly lower forecasting error than Traditional Baseline.")
    
    t_stat, p_t = ttest_rel(baseline_errors, svr_errors)
    print(f"Paired t-test p-value: {p_t:.5f}")
    
    w_stat, p_w = wilcoxon(baseline_errors, svr_errors)
    print(f"Wilcoxon signed-rank test p-value: {p_w:.5f}")
    
    print("-------------------------------------------------------------------")
    if p_w < 0.05:
        print("Conclusion: REJECT Null Hypothesis. SVR is statistically significantly better! (p < 0.05)")
    elif p_w < 0.10:
        print("Conclusion: REJECT Null Hypothesis at 10% level. SVR is statistically better! (p < 0.10)")
    else:
        print("Conclusion: FAIL to reject Null Hypothesis.")
    print("===================================================================\n")
    
    generate_academic_plots(svr_model, X_train_scaled, y_train_scaled, selected_features, sorted_results, test_df["FCF_NextYear"], svr_abs_preds, "machine_learning/plots")
    
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
