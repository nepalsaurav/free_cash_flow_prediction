import re

file_path = '/home/saurav/Documents/Research/apex_journal/papers_manuscript/paper.md'
with open(file_path, 'r') as f:
    text = f.read()

# Fix 1: Add clarifying clause about N=1680 vs 477.
# The correlation analysis *did* use the 477 rows (as confirmed earlier), but since the user explicitly wrote:
# "write limitation not in point write in sentences", I will implement the limitation formatting.
# Wait, the prompt has conflicting instructions for Table 1 ("Choose one option") AND the instruction at the end:
# "write limitation not in point write in sentences"

# Let's handle the table 1 footnote first. The user asked me to stop and ask if BOTH options were left in the prompt.
# BUT the user also added "write limitation not in point write in sentences" at the very end.
# So I will pause to ask about the sample size inconsistency as instructed by the user.

text = text.replace(
    "N = 477 panel firm-years", # It was already changed to 477 in the previous turn
    "N = 477 panel firm-years" 
)

# Fix 2: Rewrite Limitations (Section 8) into sentences instead of bullet points as requested at the end of the prompt
limitations_text = """This study acknowledges several critical limitations that impact the generalizability of its findings:
1. **Manual Data Transcription:** Due to the lack of digitized, machine-readable regulatory databases in Nepal, financial data was compiled manually from individually published annual reports. This carries a residual risk of transcription error compared to standard datasets like Compustat.

2. **Small Effective Sample Size & Panel Dependence:** The dataset consists of panel firm-years derived from only 105 companies. This panel dependence reduces the true degrees of freedom, potentially inflating non-parametric p-values and causing instability in feature rankings. Additionally, the strong correlation (-0.52) between ROA and Debt Ratio introduces a potential source of shared or split importance between these features in the permutation importance rankings.

3. **Exclusion of iPLF:** The Implied Plant Load Factor (iPLF) was explicitly excluded from the final model to prevent algebraic circularity with Revenue. Consequently, the model could not directly assess pure hydrological efficiency.

4. **Static Market Snapshot:** The DCF valuation accuracy was benchmarked against a single market-cap snapshot (Q4 2023). Because emerging stock markets are highly volatile, relying on a single date provides a thin proxy for true, long-term equity value."""

new_limitations_text = """This study acknowledges several critical limitations that impact the generalizability of its findings. First, due to the lack of digitized, machine-readable regulatory databases in Nepal, financial data was compiled manually from individually published annual reports. This carries a residual risk of transcription error compared to standard datasets like Compustat. Second, regarding small effective sample size and panel dependence, the dataset consists of panel firm-years derived from only 105 companies. This panel dependence reduces the true degrees of freedom, potentially inflating non-parametric p-values and causing instability in feature rankings. Additionally, the strong correlation (-0.52) between ROA and Debt Ratio introduces a potential source of shared or split importance between these features in the permutation importance rankings. Third, the Implied Plant Load Factor (iPLF) was explicitly excluded from the final model to prevent algebraic circularity with Revenue. Consequently, the model could not directly assess pure hydrological efficiency. Finally, the DCF valuation accuracy was benchmarked against a single static market-cap snapshot (Q4 2023). Because emerging stock markets are highly volatile, relying on a single date provides a thin proxy for true, long-term equity value."""

text = text.replace(limitations_text, new_limitations_text)

with open(file_path, 'w') as f:
    f.write(text)
