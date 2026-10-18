# NovaGen Health Risk ML — Assignment 5

## Project overview
A Streamlit application for the provided NovaGen assignment dataset. It compares Logistic Regression, Random Forest, and Gradient Boosting classifiers for the `Target` column.

## Files
- `app.py` — Streamlit app, model training, prediction, evaluation, dataset explorer.
- `novagen_dataset.csv` — dataset supplied with the assignment.
- `requirements.txt` — Python dependencies.

## Run on Windows
1. Extract the ZIP.
2. Open Command Prompt or PowerShell in this folder.
3. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   ```
4. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or run `.venv\Scripts\python.exe -m pip ...` directly.
5. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
6. Start the app:
   ```bash
   python -m streamlit run app.py
   ```

## Important notes
- The dataset provided has 9,549 rows and 23 columns, including 22 features and the `Target` label.
- `Target` contains classes 0 and 1. Follow your instructor's specification for the exact meaning of each class.
- Categorical-looking fields such as Smoking, Alcohol, Diet, and MedicalHistory are numeric codes; their exact code meanings were not fully provided, so this app does not invent those meanings.
- The app reports holdout-test metrics from a stratified 80/20 split. Results are dataset-specific and not proof of clinical validity.
- This is an educational machine-learning assignment, not a medical diagnosis or clinical decision tool.
