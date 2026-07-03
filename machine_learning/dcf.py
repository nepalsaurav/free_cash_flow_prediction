import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import os
import warnings

warnings.filterwarnings('ignore')

WACC = 0.10  # 10% Industry standard for Nepalese Hydropower
PROJECTION_YEARS = 5

def load_data():
    df = pd.read_csv("data/master_ml_dataset.csv")
    df = df.sort_values(by=["Ticker", "Year_Int"]).reset_index(drop=True)
    df["FCF_NextYear"] = df.groupby("Ticker")["OperatingCashFlow_Proxy"].shift(-1)
    df["Target_FCF_Growth"] = (df["FCF_NextYear"] - df["OperatingCashFlow_Proxy"]) / (df["OperatingCashFlow_Proxy"].abs() + 1)
    df = df.dropna(subset=["Target_FCF_Growth"]).copy()
    lower = df["Target_FCF_Growth"].quantile(0.05)
    upper = df["Target_FCF_Growth"].quantile(0.95)
    df["Target_FCF_Growth"] = df["Target_FCF_Growth"].clip(lower, upper)
    df = df.fillna(0)
    return df

def get_trained_svr(df, features):
    train_df = df[df["Year_Int"] < 2023]
    X_train = train_df[features]
    y_train = train_df["Target_FCF_Growth"]
    
    # Skip Feature Selection (SVR Kernel handles the 14D space)
    X_train_sel = X_train
    selected_features = features
    selector = None
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    
    # Train Best SVR
    svr_search = RandomizedSearchCV(SVR(), {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.1, 0.01], "epsilon": [0.01, 0.1, 1]}, n_iter=15, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1, random_state=42)
    svr_search.fit(X_train_scaled, y_train)
    svr_model = svr_search.best_estimator_
    
    return svr_model, selector, scaler, selected_features

def run_dcf():
    features = [
        "Revenue", "Debt_Ratio", "ROA", "Working_Capital", "CapEx",
        "Installed_Capacity_MW", "PLF", "License_Term_Years",
        "GDP_Growth", "Inflation", "Interest_Rate", 
        "Years_Since_COD", "Pre_Post_COD", "EBIT_Margin"
    ]
    
    df = load_data()
    svr_model, selector, scaler, selected_features = get_trained_svr(df, features)
    
    # We will do DCF valuation for the test set (2023 data)
    test_df = df[df["Year_Int"] >= 2023].copy()
    
    results = []
    
    for _, row in test_df.iterrows():
        ticker = row["Ticker"]
        base_fcf = row["OperatingCashFlow_Proxy"]
        license_remaining = row["License_Term_Years"]
        
        # Real variables extracted from hydro.csv
        actual_market_cap = row["Market_Cap"]
        total_debt = row["Total_Debt"]
        
        # 1. Traditional Forecasting (Assume 0% growth, so cash flow stays flat)
        trad_forecasts = [base_fcf for _ in range(PROJECTION_YEARS)]
        
        # 2. ML Forecasting (Iterative)
        ml_forecasts = []
        current_fcf = base_fcf
        
        # Get the feature row for prediction
        x_raw = pd.DataFrame([row[features]])
        x_sel = x_raw
        x_scaled = scaler.transform(x_sel)
        
        pred_growth = svr_model.predict(x_scaled)[0]
        
        for _ in range(PROJECTION_YEARS):
            next_fcf = current_fcf + (np.abs(current_fcf) + 1) * pred_growth
            ml_forecasts.append(next_fcf)
            current_fcf = next_fcf
            
        # 3. Discount Cash Flows
        trad_pv = sum(fcf / ((1 + WACC) ** (t + 1)) for t, fcf in enumerate(trad_forecasts))
        ml_pv = sum(fcf / ((1 + WACC) ** (t + 1)) for t, fcf in enumerate(ml_forecasts))
        
        # 4. Finite-Horizon Terminal Value (TV)
        remaining_years_after_forecast = max(0, int(license_remaining) - PROJECTION_YEARS)
        
        trad_tv_pv = 0
        ml_tv_pv = 0
        if remaining_years_after_forecast > 0:
            trad_tv = sum(trad_forecasts[-1] / ((1 + WACC) ** t) for t in range(1, remaining_years_after_forecast + 1))
            ml_tv = sum(ml_forecasts[-1] / ((1 + WACC) ** t) for t in range(1, remaining_years_after_forecast + 1))
            
            # Discount TV back to present (Year 5)
            trad_tv_pv = trad_tv / ((1 + WACC) ** PROJECTION_YEARS)
            ml_tv_pv = ml_tv / ((1 + WACC) ** PROJECTION_YEARS)
            
        trad_enterprise_value = trad_pv + trad_tv_pv
        ml_enterprise_value = ml_pv + ml_tv_pv
        
        # 5. Calculate EQUITY VALUE (Enterprise Value - Total Debt)
        trad_equity_value = trad_enterprise_value - total_debt
        ml_equity_value = ml_enterprise_value - total_debt
        
        # Ensure Equity Value isn't wildly negative for comparison purposes
        trad_equity_value = max(0, trad_equity_value)
        ml_equity_value = max(0, ml_equity_value)
        
        # 6. Compare to REAL Market Cap
        trad_error = np.abs(actual_market_cap - trad_equity_value)
        ml_error = np.abs(actual_market_cap - ml_equity_value)
        
        results.append({
            "Ticker": ticker,
            "Traditional_EV": trad_enterprise_value,
            "ML_EV": ml_enterprise_value,
            "Total_Debt": total_debt,
            "Trad_Equity": trad_equity_value,
            "ML_Equity": ml_equity_value,
            "Real_Market_Cap": actual_market_cap,
            "Trad_Error": trad_error,
            "ML_Error": ml_error
        })
        
    res_df = pd.DataFrame(results)
    
    print("\n===================================================================")
    print("                  SECTION 8.8: TRUE DCF VALUATION STAGE")
    print("===================================================================")
    print(f"Total Companies Valued: {len(res_df)}")
    print(f"WACC Applied: {WACC*100}%")
    print("Terminal Value Methodology: Finite-Horizon (Bounded by PPA License)")
    print("Market Cap & Debt Source: Actual Q4 figures from NEPSE/hydro.csv")
    
    avg_trad_error = res_df["Trad_Error"].mean()
    avg_ml_error = res_df["ML_Error"].mean()
    
    improvement = ((avg_trad_error - avg_ml_error) / avg_trad_error) * 100
    
    print("\n[TRUE VALUATION ACCURACY vs NEPSE MARKET CAP]")
    print(f"Average Traditional Valuation Error: Rs {avg_trad_error:,.0f}")
    print(f"Average Machine Learning (SVR) Valuation Error: Rs {avg_ml_error:,.0f}")
    print("-------------------------------------------------------------------")
    if avg_ml_error < avg_trad_error:
        print(f"EMPIRICAL FINDING FOR RQ3: Machine Learning (SVR) produces intrinsically more accurate")
        print(f"Enterprise Valuations than Traditional DCF models by a margin of {improvement:.2f}%.")
    else:
        print(f"EMPIRICAL FINDING FOR RQ3: Traditional DCF models performed better by {-improvement:.2f}%.")
        
    print("===================================================================\n")

if __name__ == "__main__":
    run_dcf()
