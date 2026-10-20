# Utorogu Production Intelligence V3

Responsive Streamlit application for field- and completion-level production exploration, fixed-origin forecasting, rolling-origin validation and model comparison.

## Run on Windows / VS Code

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a **private** GitHub repository and upload `app.py`, `requirements.txt`, and `production.xlsx` from this folder.
2. Sign in to https://share.streamlit.io/ and create an app from that repository, with `app.py` as the entry point.
3. Be aware that a publicly accessible Streamlit deployment exposes dashboard content and potentially dataset-derived information to anyone with access to the app. Check your rights to publish the production data before deploying.

## Research caveats

- Units and whether the source contains monthly totals or rates have not been independently established.
- Scenario bands are ±20% sensitivities, **not** statistical prediction intervals.
- Long-range forecasts are extrapolations and not reserves estimates.
- Models are refitted from history available at each cutoff; future actuals are used for scoring only.
- Model rankings depend on selected origins, horizon, stream and available actuals.
