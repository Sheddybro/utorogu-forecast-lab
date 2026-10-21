"""Offline, reproducible V5 checks. Run: python verify_project.py"""
from pathlib import Path
import numpy as np
import pandas as pd
from data_quality import clean_production
from models import forecast, metrics, MODELS

root=Path(__file__).resolve().parent
raw=pd.read_excel(root/'production.xlsx',sheet_name='UTOROGU')
assert len(raw)==13587, f'Unexpected raw count: {len(raw)}'
clean,audit=clean_production(raw,end_date='2014-12-01')
shares=clean.groupby('UNIQUEID')['OIL'].apply(lambda s:(s.fillna(0)>0).mean())
selected=clean[clean.UNIQUEID.isin(shares[shares>=.40].index)].copy()
assert selected.UNIQUEID.nunique()==47, 'Completion count does not match notebook selection'
assert len(selected)==10441, f'Unexpected selected rows: {len(selected)}'
assert selected.DATE.max()==pd.Timestamp('2014-12-01')
monthly=selected.groupby('DATE')['OIL'].sum(min_count=1).sort_index().asfreq('MS')
future=pd.date_range('2015-01-01',periods=12,freq='MS')
for name in ['Seasonal Naive','Arps Exponential DCA','Linear Trend','Random Forest','XGBoost']:
    pred=forecast(monthly,future,name)
    assert len(pred)==12 and np.isfinite(pred).all() and (pred>=0).all(),name
    print('PASS',name,'12 months')
print('PASS raw rows',len(raw),'selected entities',selected.UNIQUEID.nunique(),'selected rows',len(selected))
print('NOT VERIFIED: Prophet, LSTM, hybrid, ensembles, UI, cloud deployment, notebook numerical equivalence')
