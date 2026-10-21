# Utorogu Production Intelligence — V6

## Run locally

Use Python 3.11 or 3.12. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501. Use the sidebar expand arrow to show forecasting controls.

## Deploy on Streamlit Community Cloud

Push the contents of **this folder** to a GitHub repository. Set **branch** to `main` (if that is your Git branch), **main file** to `app.py`. The production.xlsx workbook must be present alongside app.py, or the app will not start. Check rights and confidentiality before publishing the workbook. Heavy models such as TensorFlow/LSTM and Prophet may exceed free hosting resources. The `requirements.txt` lists optional heavy dependencies; a cloud deployment may need additional resources or a separate worker to run them reliably.

## Changes from V5

- Compact collapsible sidebar (250–290 px) with advanced cleaning controls in an expander.
- Full-width charts, compact hero, readable KPI metrics and uncluttered single-page workspace navigation.
- Chart axis title removed to avoid unreadable rotated text in narrow views.
- Retrospective model comparison restricted to December-origin tests whose forecast horizons fall inside 2006–2014; results are chronological and fixed-origin.
- Relative production workbook path (robust to launch working directory).

## Research and validation limitations

- The workbook has 13,587 raw records. Notebook-style 40% positive-oil-month filter through December 2014 yields 47 completions and 10,441 cleaned rows under the supplied cleaning implementation.
- Models offered: Seasonal Naive, Arps Exponential DCA, Linear Trend, Random Forest, Prophet, XGBoost, LSTM, Prophet + XGBoost Hybrid, Equal-weight Ensemble, Validation-weighted Ensemble.
- Five models (Seasonal Naive, Arps, Linear Trend, Random Forest, XGBoost) have automated local numerical smoke tests. Prophet, LSTM, hybrid, and ensembles have NOT been runtime-verified in this environment; model results have NOT been matched to the notebook's reported numbers.
- The current XGBoost and LSTM feature/architecture choices are simplified and are NOT exact notebook replication. Missing observations and unknown production units require interpretation. ±20% is sensitivity, not a confidence interval.
- Browser rendering and Streamlit Cloud deployment have NOT been directly verified.

Run `python verify_project.py` to repeat the offline dataset and five-model smoke tests.
