"""Leakage-aware fixed-origin forecast engines. No future actuals enter prediction features."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

MODELS=['Seasonal Naive','Arps Exponential DCA','Linear Trend','Random Forest','Prophet','XGBoost','LSTM','Prophet + XGBoost Hybrid','Equal-weight Ensemble','Validation-weighted Ensemble']
BASE=['Seasonal Naive','Arps Exponential DCA','Prophet','XGBoost','LSTM']

def features(dates,origin):
    t=np.array([(d.year-origin.year)*12+d.month-origin.month for d in dates],dtype=float)
    m=np.array([d.month for d in dates]);return np.column_stack([t,np.sin(2*np.pi*m/12),np.cos(2*np.pi*m/12),np.maximum(t,0)**.5])

def clean_history(series):
    s=series.astype(float).sort_index()
    if len(s)<36:raise ValueError('At least 36 monthly training observations required.')
    s=s.interpolate(method='time',limit_direction='both')
    if s.isna().any():raise ValueError('Unresolved missing training observations.')
    return s.clip(lower=0)

def _arps(h,dates):
    a=h.to_numpy(); peak=int(np.argmax(a)); tail=a[peak:]
    if len(tail)<6:tail=a
    x=np.arange(len(tail),dtype=float)
    def decline(t,qi,D):return qi*np.exp(-D*t)
    try:qi,D=curve_fit(decline,x,tail,p0=[max(tail[0],1e-6),.01],bounds=([0,0],[np.inf,1]),maxfev=15000)[0]
    except Exception:qi,D=max(float(tail[-1]),0),.01
    step=np.arange(len(tail),len(tail)+len(dates),dtype=float)
    return decline(step,qi,D)

def _prophet(h,dates):
    from prophet import Prophet
    train=pd.DataFrame({'ds':h.index,'y':h.values})
    mode='multiplicative' if (h>0).all() else 'additive'
    model=Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False,seasonality_mode=mode)
    model.fit(train)
    return model,np.asarray(model.predict(pd.DataFrame({'ds':dates}))['yhat'],dtype=float)

def _xgb(h,dates,residual=None):
    from xgboost import XGBRegressor
    x=features(h.index,h.index[0]);xf=features(dates,h.index[0])
    y=h.to_numpy() if residual is None else residual
    m=XGBRegressor(n_estimators=220,max_depth=3,learning_rate=.035,subsample=.9,colsample_bytree=.9,random_state=42,n_jobs=2,objective='reg:squarederror')
    m.fit(x,y)
    return np.asarray(m.predict(xf),dtype=float)

def _lstm(h,dates):
    # Recursive autoregression: future values are generated from past predictions only.
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    tf.keras.utils.set_random_seed(42)
    seq=12; values=h.to_numpy().reshape(-1,1)
    scaler=StandardScaler().fit(values)
    z=scaler.transform(values).ravel()
    X=np.array([z[i-seq:i] for i in range(seq,len(z))])[:,:,None]
    y=z[seq:]
    tf.keras.backend.clear_session()
    model=tf.keras.Sequential([tf.keras.layers.Input((seq,1)),tf.keras.layers.LSTM(24),tf.keras.layers.Dense(1)])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X,y,epochs=25,batch_size=24,verbose=0,shuffle=False)
    window=list(z[-seq:]); out=[]
    for _ in dates:
        p=float(model.predict(np.array(window[-seq:])[None,:,None],verbose=0)[0,0]);out.append(p);window.append(p)
    return scaler.inverse_transform(np.array(out).reshape(-1,1)).ravel()

def forecast(history,dates,name,calibrate=True):
    h=clean_history(history);dates=pd.DatetimeIndex(dates)
    if not len(dates):return np.array([])
    if name=='Seasonal Naive':
        last={d.month:float(v) for d,v in h.iloc[-12:].items()}
        p=np.array([last.get(d.month,float(h.iloc[-1])) for d in dates])
    elif name=='Arps Exponential DCA':p=_arps(h,dates)
    elif name=='Prophet':p=_prophet(h,dates)[1]
    elif name=='XGBoost':p=_xgb(h,dates)
    elif name=='LSTM':p=_lstm(h,dates)
    elif name=='Prophet + XGBoost Hybrid':
        pm,pred=_prophet(h,dates)
        fitted=pm.predict(pd.DataFrame({'ds':h.index}))['yhat'].to_numpy()
        p=pred+_xgb(h,dates,residual=h.to_numpy()-fitted)
    elif name=='Linear Trend':
        m=LinearRegression().fit(features(h.index,h.index[0]),h.values)
        p=m.predict(features(dates,h.index[0]))
    elif name=='Random Forest':
        m=RandomForestRegressor(n_estimators=150,min_samples_leaf=3,random_state=42,n_jobs=-1).fit(features(h.index,h.index[0]),h.values)
        p=m.predict(features(dates,h.index[0]))
    elif name in ('Equal-weight Ensemble','Validation-weighted Ensemble'):
        components=['Seasonal Naive','Arps Exponential DCA','Prophet','XGBoost']
        preds=np.stack([forecast(h,dates,m,False) for m in components])
        weights=np.ones(len(components))
        if name=='Validation-weighted Ensemble' and calibrate and len(h)>72:
            # Internal calibration uses ONLY months before the forecast origin.
            cut=24; inner=h.iloc[:-cut]; val_dates=h.index[-cut:]
            for i,m in enumerate(components):
                try:
                    pp=forecast(inner,val_dates,m,False)
                    weights[i]=1/(np.mean(np.abs(pp-h.iloc[-cut:].values))+1e-6)
                except Exception:weights[i]=0
        weights=weights/weights.sum() if weights.sum()>0 else np.ones(len(components))/len(components)
        p=np.average(preds,axis=0,weights=weights)
    else:raise ValueError('Unknown model: '+name)
    return np.maximum(0,np.asarray(p,dtype=float))

def metrics(actual,pred):
    a=np.asarray(actual,dtype=float);p=np.asarray(pred,dtype=float)
    ok=np.isfinite(a)&np.isfinite(p)
    if not ok.any():return {'MAE':np.nan,'RMSE':np.nan,'sMAPE (%)':np.nan,'Observed':0}
    a=a[ok];p=p[ok];den=np.abs(a)+np.abs(p)
    sm=np.where(den>0,2*np.abs(a-p)/np.maximum(den,1e-12),0)
    return {'MAE':float(np.mean(np.abs(a-p))),'RMSE':float(np.sqrt(np.mean((a-p)**2))),'sMAPE (%)':float(np.mean(sm)*100),'Observed':len(a)}
