import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Replicate data prep
df = pd.read_csv('/home/saurav/Documents/Research/apex_journal/data/master_ml_dataset.csv')
features = ["Revenue", "Debt_Ratio", "ROA", "Working_Capital", "CapEx",
            "Installed_Capacity_MW", "License_Term_Years",
            "GDP_Growth", "Inflation", "Interest_Rate", 
            "Years_Since_COD", "Pre_Post_COD", "EBIT_Margin"]
df["FCF_NextYear"] = df.groupby("Ticker")["OperatingCashFlow_Proxy"].shift(-1)
df["Target_FCF_Growth"] = (df["FCF_NextYear"] - df["OperatingCashFlow_Proxy"]) / (df["OperatingCashFlow_Proxy"].abs() + 1)
df = df.dropna(subset=["Target_FCF_Growth"]).copy()
lower, upper = df["Target_FCF_Growth"].quantile(0.05), df["Target_FCF_Growth"].quantile(0.95)
df["Target_FCF_Growth"] = df["Target_FCF_Growth"].clip(lower, upper)
df = df.fillna(0)
df["Year"] = df["Year"].astype(str).str[:4].astype(int)

train_df = df[df["Year"] <= 2024]
test_df = df[df["Year"] == 2025]
X_train = train_df[features]
y_train = train_df["Target_FCF_Growth"]
X_test = test_df[features]
y_test = test_df["Target_FCF_Growth"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr_model = SVR(C=1.0, epsilon=0.1) 
svr_model.fit(X_train_scaled, y_train)
pred_growth = svr_model.predict(X_test_scaled)
mae_growth = mean_absolute_error(y_test, pred_growth)

print(f"SVR MAE in Target Growth Rate space: {mae_growth:.4f}")

trad_growth = np.zeros(len(y_test)) # Traditional is 0% growth
trad_mae_growth = mean_absolute_error(y_test, trad_growth)
print(f"Traditional Baseline MAE in Target Growth Rate space: {trad_mae_growth:.4f}")
