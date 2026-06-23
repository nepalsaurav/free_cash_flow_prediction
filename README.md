# Machine Learning Free Cash Flow Forecasting for Hydropower

Welcome to the data engineering and machine learning repository for the Nepalese Hydropower Intrinsic Valuation research project.

This repository contains the complete pipeline for extracting, transforming, and modeling 16 years of financial, operational, and macroeconomic data for 105 Hydropower companies.

---

## 🛠 Prerequisites & Installation (Windows)

This project strictly uses **`uv`** (by Astral) for extremely fast Python package management and virtual environment execution. You do not need to install `pip` or manage virtual environments manually.

### 1. Install `uv` on Windows
Open your Windows **PowerShell** and run the following command:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
*(If you are on macOS or Linux, use: `curl -LsSf https://astral.sh/uv/install.sh | sh`)*

### 2. Verify Installation
Restart your terminal and verify it is installed by running:
```bash
uv --version
```

### 3. Setup the Project
Navigate into this project directory. You do **not** need to manually install dependencies from a `requirements.txt`. `uv` will automatically read the `pyproject.toml` and `uv.lock` files to set up the exact Python environment instantly.

To synchronize your environment:
```bash
uv sync
```

---

## 🚀 How to Run Scripts

Instead of activating a virtual environment or running `python script.py`, use the `uv run` command. This ensures the script is executed in the perfectly isolated project environment.

**Example: Re-running the dataset merge script:**
```bash
uv run scripts/merge_master_dataset.py
```
*(If a script requires specific packages like pandas, `uv run` automatically handles it!)*

---

## 📂 Repository Structure

*   **`data/`**: Contains the core datasets. `master_ml_dataset.csv` is the final merged 16-year panel dataset ready for modeling.
*   **`scripts/`**: Contains all the Python scripts used to scrape, extract, impute, and merge the datasets.
*   **`machine_learning/`**: Contains the specific guide (`ML_COAUTHOR_GUIDE.md`) and future Jupyter Notebooks for the Data Science team.
*   **`data_reviewer/`**: Contains the `DATA_VALIDATION_GUIDE.md` instructing co-authors on how to audit the dataset for purity against audited Annual Reports.
*   **`progress_track/`**: Contains explicit methodology logs and data dictionaries for the research paper.
*   **`misc_data/`, `nea_annual_reports/`, `nrb_reports/`**: The raw source files (PDFs, CSVs, Excel) used for extraction.

---

## 👥 Co-Author Guides
*   **Data Reviewers**: Start by reading `data_reviewer/DATA_VALIDATION_GUIDE.md`.
*   **Machine Learning Team**: Start by reading `machine_learning/ML_COAUTHOR_GUIDE.md` for modeling objectives and baseline DCF requirements.
