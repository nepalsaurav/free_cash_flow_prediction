import re

file_path = '/home/saurav/Documents/Research/apex_journal/papers_manuscript/paper.md'
with open(file_path, 'r') as f:
    text = f.read()

# Fix 1: N=1680 to N=477
text = text.replace(
    "N = 1680 panel firm-years",
    "N = 477 panel firm-years"
)

# Fix 2: Remove redundant "Figure N:"
text = re.sub(r'!\[Figure \d+: ', '![', text)

# Fix 3: Convert Section 8 (Limitations) to vertical numbered list by adding newlines
limitations_text = """1. **Manual Data Transcription:** Due to the lack of digitized, machine-readable regulatory databases in Nepal, financial data was compiled manually from individually published annual reports. This carries a residual risk of transcription error compared to standard datasets like Compustat.
2. **Small Effective Sample Size & Panel Dependence:** The dataset consists of panel firm-years derived from only 105 companies. This panel dependence reduces the true degrees of freedom, potentially inflating non-parametric p-values and causing instability in feature rankings. Additionally, the strong correlation (-0.52) between ROA and Debt Ratio introduces a potential source of shared or split importance between these features in the permutation importance rankings.
3. **Exclusion of iPLF:** The Implied Plant Load Factor (iPLF) was explicitly excluded from the final model to prevent algebraic circularity with Revenue. Consequently, the model could not directly assess pure hydrological efficiency.
4. **Static Market Snapshot:** The DCF valuation accuracy was benchmarked against a single market-cap snapshot (Q4 2023). Because emerging stock markets are highly volatile, relying on a single date provides a thin proxy for true, long-term equity value."""

new_limitations_text = """
1. **Manual Data Transcription:** Due to the lack of digitized, machine-readable regulatory databases in Nepal, financial data was compiled manually from individually published annual reports. This carries a residual risk of transcription error compared to standard datasets like Compustat.

2. **Small Effective Sample Size & Panel Dependence:** The dataset consists of panel firm-years derived from only 105 companies. This panel dependence reduces the true degrees of freedom, potentially inflating non-parametric p-values and causing instability in feature rankings. Additionally, the strong correlation (-0.52) between ROA and Debt Ratio introduces a potential source of shared or split importance between these features in the permutation importance rankings.

3. **Exclusion of iPLF:** The Implied Plant Load Factor (iPLF) was explicitly excluded from the final model to prevent algebraic circularity with Revenue. Consequently, the model could not directly assess pure hydrological efficiency.

4. **Static Market Snapshot:** The DCF valuation accuracy was benchmarked against a single market-cap snapshot (Q4 2023). Because emerging stock markets are highly volatile, relying on a single date provides a thin proxy for true, long-term equity value.
"""
text = text.replace(limitations_text, new_limitations_text.strip() + "\n")

# Fix 4: Convert Section 7 results into proper bullet points
results_text = """The empirical results were:
*   The **Traditional DCF** Valuation yielded an average error of **Rs 5.73 Million** per company.
*   The **SVR-driven DCF** Valuation yielded an average error of **Rs 5.51 Million** per company."""

new_results_text = """The empirical results were:

- The **Traditional DCF** Valuation yielded an average error of **Rs 5.73 Million** per company.
- The **SVR-driven DCF** Valuation yielded an average error of **Rs 5.51 Million** per company."""
text = text.replace(results_text, new_results_text)

with open(file_path, 'w') as f:
    f.write(text)
