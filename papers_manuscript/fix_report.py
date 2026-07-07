import re

file_path = '/home/saurav/Documents/Research/apex_journal/papers_manuscript/paper.md'
with open(file_path, 'r') as f:
    text = f.read()

# Fix 1: Insert target_distribution.png (Figure 1)
target_distribution_insert = """
This growth metric was subsequently Winsorized at the 5th and 95th percentiles to mitigate mathematical explosions caused by near-zero denominators.

![Figure 1: Distribution of Target Variable (Free Cash Flow Growth)](../plots/target_distribution.png)

The non-normal distribution (heavy skew and fat tails) of Free Cash Flow mathematically justified exploring non-linear forecasting algorithms.
"""
text = text.replace(
    "This growth metric was subsequently Winsorized at the 5th and 95th percentiles to mitigate mathematical explosions caused by near-zero denominators. The non-normal distribution (heavy skew and fat tails) of Free Cash Flow mathematically justified exploring non-linear forecasting algorithms.",
    target_distribution_insert.strip()
)

# Fix 2: Add Figure 2 to model_comparison
text = text.replace(
    "![Machine Learning Algorithm Performance vs Traditional Baseline](../plots/model_comparison.png)",
    "![Figure 2: Machine Learning Algorithm Performance vs Traditional Baseline](../plots/model_comparison.png)"
)

# Fix 3: Insert actual_vs_predicted.png (Figure 3)
actual_vs_predicted_insert = """
Nonetheless, we find modest evidence that SVR provides a more robust mechanism for forecasting fundamental cash flows compared to static baseline assumptions.

![Figure 3: SVR Validation: Actual vs. Predicted Free Cash Flow (Out-of-Sample)](../plots/actual_vs_predicted.png)
"""
text = text.replace(
    "Nonetheless, we find modest evidence that SVR provides a more robust mechanism for forecasting fundamental cash flows compared to static baseline assumptions.",
    actual_vs_predicted_insert.strip()
)

# Fix 4: Add Figure 4 to learning curve
text = text.replace(
    "![SVR Learning Curve (GroupKFold)](../plots/learning_curve.png)",
    "![Figure 4: SVR Learning Curve (GroupKFold)](../plots/learning_curve.png)"
)

# Fix 5: Add Figure 5 to feature importance
text = text.replace(
    "![Permutation Importance inside the SVR Kernel ($n = 30$, $\\pm$ SD)](../plots/feature_importance.png)",
    "![Figure 5: Permutation Importance inside the SVR Kernel ($n = 30$, $\\pm$ SD)](../plots/feature_importance.png)"
)

# Fix 6: Insert pdp_key_features.png (Figure 6)
pdp_insert = """
To move beyond simply identifying *which* variables mattered, Partial Dependence Plots (PDP) (Friedman, 2001) were generated to map how the SVR model interpreted these structural constraints.

![Figure 6: Partial Dependence of FCF Growth on Key Economic Drivers](../plots/pdp_key_features.png)

The interpretability of the SVR model yielded several critical insights into the unique economics of the Nepalese Hydropower sector:
"""
text = text.replace(
    "To move beyond simply identifying *which* variables mattered, Partial Dependence Plots (PDP) (Friedman, 2001) were generated to map how the SVR model interpreted these structural constraints.\n\nThe interpretability of the SVR model yielded several critical insights into the unique economics of the Nepalese Hydropower sector:",
    pdp_insert.strip()
)

# Fix 7: Add Figure 7 to valuation comparison
text = text.replace(
    "![Intrinsic Valuation vs Actual Market Cap (Top 5 Companies)](../plots/valuation_comparison.png)",
    "![Figure 7: Intrinsic Valuation vs Actual Market Cap (Top 5 Companies)](../plots/valuation_comparison.png)"
)

# Fix 8: Correct the correlation significance note
text = text.replace(
    "*Note. * indicates statistical significance at standard levels (absolute value > 0.05). Upper triangle omitted for clarity.*",
    "*Note. * indicates $p < 0.05$ based on a two-tailed test of the Pearson correlation coefficient ($N = 1680$ panel firm-years). Upper triangle omitted for clarity.*"
)

# Fix 9: Add sentence to Limitations
limitations_add = """1. **Manual Data Transcription:** Due to the lack of digitized, machine-readable regulatory databases in Nepal, financial data was compiled manually from individually published annual reports. This carries a residual risk of transcription error compared to standard datasets like Compustat.
2. **Small Effective Sample Size & Panel Dependence:** The dataset consists of panel firm-years derived from only 105 companies. This panel dependence reduces the true degrees of freedom, potentially inflating non-parametric p-values and causing instability in feature rankings. Additionally, the strong correlation (-0.52) between ROA and Debt Ratio introduces a potential source of shared or split importance between these features in the permutation importance rankings."""
text = text.replace(
    "1. **Manual Data Transcription:** Due to the lack of digitized, machine-readable regulatory databases in Nepal, financial data was compiled manually from individually published annual reports. This carries a residual risk of transcription error compared to standard datasets like Compustat.\n2. **Small Effective Sample Size & Panel Dependence:** The dataset consists of panel firm-years derived from only 105 companies. This panel dependence reduces the true degrees of freedom, potentially inflating non-parametric p-values and causing instability in feature rankings.",
    limitations_add
)

with open(file_path, 'w') as f:
    f.write(text)
