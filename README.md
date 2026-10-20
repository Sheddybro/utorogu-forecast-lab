# Utorogu Production Intelligence V5

A Streamlit forecasting and data-quality application based on the Utorogu OML 34 spreadsheet and the accompanying thesis/notebook.

## Start locally

Python 3.11 or 3.12 recommended (64-bit).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Free cloud deployment

Commit `app.py`, `models.py`, `data_quality.py`, `production.xlsx`, and `requirements.txt` to a GitHub repository. Set the Streamlit Community Cloud entry point to `app.py` and select Python 3.11/3.12 if offered. **TensorFlow and Prophet make the installation larger and LSTM/ensemble runs slower**; free resource limits may be exceeded. If the deployment fails because of memory, use a higher-memory hosting service or omit the optional LSTM dependency and do not select LSTM models.

## Research caveats

This dashboard implements the named model families but does not claim to exactly reproduce the notebook's hyperparameters or its feature-based multivariate ML results. In particular XGBoost is calendar-feature based, LSTM is recursive univariate, and Prophet+XGBoost uses a calendar-feature residual learner. Validation is fixed-origin; do not use future actuals as predictors. The notebook uses an entity positive-oil-month share rule rather than the observed-span coverage filter of earlier dashboard versions. With the bundled data, December 2014 cutoff and 40% positive-oil-month rule, the cleaning implementation retained 47 entities in a local verification. This matches the thesis count but does not prove that every individual entity or downstream forecast is identical; compare entity IDs and outputs against the notebook. Units are unverified; forecast bands are sensitivity scenarios.
