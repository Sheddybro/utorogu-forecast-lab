"""Auditable production cleaning: never silently invent zeros or remove outliers."""
import pandas as pd
import numpy as np

STREAMS=['OIL','GAS','WATER','COND']

def clean_production(source, duplicate_policy='sum', negative_policy='flag', coverage_threshold=0.0, coverage_stream='OIL', end_date=None):
    d=source.copy(); audit=[]
    def log(rule,count,action): audit.append({'Check':rule,'Affected':int(count),'Action':action})
    d.columns=[str(c).strip().upper() for c in d.columns]
    required=['UNIQUEID','DATE','OIL','GAS','WATER']
    missing=[x for x in required if x not in d]
    if missing:raise ValueError(f'Missing columns: {missing}')
    for col in STREAMS+['DAYS']:
        if col not in d:d[col]=np.nan
        original=d[col].copy()
        d[col]=pd.to_numeric(d[col],errors='coerce')
        log(f'Non-numeric {col}',(original.notna()&d[col].isna()).sum(),'Converted to missing; no imputation')
    d['DATE']=pd.to_datetime(d['DATE'],errors='coerce')
    bad=d.DATE.isna()|d.UNIQUEID.isna()|d.UNIQUEID.astype(str).str.strip().eq('')
    log('Invalid date or completion ID',bad.sum(),'Rows excluded')
    d=d.loc[~bad].copy()
    d['DATE']=d.DATE.dt.to_period('M').dt.to_timestamp()
    if end_date is not None:
        excluded=(d.DATE>pd.Timestamp(end_date)).sum()
        log('After study end date',excluded,'Excluded for thesis-aligned analysis')
        d=d[d.DATE<=pd.Timestamp(end_date)].copy()
    dup=d.duplicated(['UNIQUEID','DATE'],keep=False)
    log('Repeated completion-month rows',dup.sum(),'Aggregated by sum' if duplicate_policy=='sum' else 'Kept first row')
    if duplicate_policy=='first':d=d.drop_duplicates(['UNIQUEID','DATE'],keep='first')
    else:d=d.groupby(['UNIQUEID','DATE'],as_index=False)[STREAMS+['DAYS']].sum(min_count=1)
    for col in STREAMS:
        neg=d[col]<0
        log(f'Negative {col}',neg.sum(),'Set to missing' if negative_policy=='missing' else 'Flagged and retained')
        if negative_policy=='missing':d.loc[neg,col]=np.nan
    if coverage_threshold>0 and not d.empty:
        # Coverage is assessed over each entity's observed start/end span, not whole field lifetime.
        coverage={}
        for uid,g in d.groupby('UNIQUEID'):
            span=(g.DATE.max().year-g.DATE.min().year)*12+g.DATE.max().month-g.DATE.min().month+1
            coverage[uid]=g[coverage_stream].notna().sum()/max(span,1)
        retained=[uid for uid,v in coverage.items() if v>=coverage_threshold]
        log('Completion coverage filter',d.loc[~d.UNIQUEID.isin(retained),'UNIQUEID'].nunique(),f'Excluded entities below {coverage_threshold:.0%} {coverage_stream} coverage')
        d=d[d.UNIQUEID.isin(retained)].copy()
    else:log('Completion coverage filter',0,'No filter')
    for col in STREAMS:
        s=d[col].dropna()
        if len(s)>10:
            q1,q3=s.quantile([.25,.75]); upper=q3+3*(q3-q1)
            log(f'Extreme {col} (3×IQR)',(d[col]>upper).sum(),'Flag only; no automatic clipping')
    return d.sort_values(['DATE','UNIQUEID']).reset_index(drop=True),pd.DataFrame(audit)
