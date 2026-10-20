import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_quality import clean_production
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title='Utorogu | Production Intelligence', page_icon='◈', layout='wide', initial_sidebar_state='expanded')
st.markdown("""<style>
    .stApp {
        background: linear-gradient(180deg, #061b2d 0%, #091d2e 100%);
        color: #f3f8ff;
    }
    .block-container {
        padding-top: 0.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #091e2d 0%, #0f2337 100%);
        border-right: 1px solid rgba(120, 152, 180, 0.35);
        padding: 0.6rem 0.9rem 0.8rem;
    }
    section[data-testid="stSidebar"] * {
        color: #edf6ff !important;
    }
    section[data-testid="stSidebar"] .stSubheader {
        margin-top: 0.8rem;
        padding: 0.15rem 0;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        color: #f6fbff !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #ebf5ff !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stRadio [role="radio"],
    section[data-testid="stSidebar"] .stCheckbox [role="checkbox"] {
        background: transparent !important;
        border-color: rgba(255,255,255,0.7) !important;
    }
    section[data-testid="stSidebar"] .stRadio [role="radio"]::before,
    section[data-testid="stSidebar"] .stCheckbox [role="checkbox"]::before {
        background: #ff5a5a !important;
        border-color: #ff5a5a !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stNumberInput > div > div,
    section[data-testid="stSidebar"] .stTextInput > div > div,
    section[data-testid="stSidebar"] .stDateInput > div > div,
    section[data-testid="stSidebar"] .stSlider [data-testid="stBaseSlider"] {
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        border-radius: 12px !important;
        color: #122a3d !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.26) !important;
        border-radius: 10px !important;
        color: #edf7ff !important;
        font-weight: 700 !important;
        padding: 0.6rem 0.8rem !important;
        width: 100% !important;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(180deg, #1f5d88 0%, #103b5a 100%) !important;
        color: #edf9ff !important;
        border: 1px solid rgba(122, 172, 206, 0.65) !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetric"] {
        background: rgba(20, 42, 58, 0.92);
        border: 1px solid rgba(116, 153, 181, 0.42);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 118px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stMetric"] * {
        color: #f3f8ff !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #cfe1f0 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.8rem, 2vw, 2.7rem) !important;
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
    }
    .stTabs [role="tablist"] {
        gap: 1.2rem;
        border-bottom: 1px solid rgba(142, 170, 195, 0.3);
        margin-top: 1.3rem;
    }
    .stTabs [role="tab"] {
        color: #d9e8f5 !important;
        font-weight: 500 !important;
        padding: 0.55rem 0.15rem 0.8rem !important;
        border-bottom: 2px solid transparent;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        color: #ff6d67 !important;
        border-bottom-color: #ff6d67 !important;
    }
    .stCaption {
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: rgba(195, 214, 230, 0.72) !important;
        font-size: 0.7rem !important;
    }
    h1 {
        font-size: clamp(2.8rem, 4vw, 4.2rem) !important;
        letter-spacing: -0.07em !important;
        line-height: 1.1 !important;
        margin: 0.1rem 0 0.8rem !important;
        color: #f5faff !important;
        font-weight: 800 !important;
    }
    h2, h3, h4, h5, h6 {
        color: #f5fbff !important;
    }
    .stAlert {
        background: rgba(17, 39, 52, 0.7) !important;
        border: 1px solid rgba(116, 152, 179, 0.35) !important;
    }
    .stDataFrame {
        background: rgba(12, 27, 38, 0.65);
        border-radius: 12px;
    }
    .stPlotlyChart > div {
        border-radius: 12px;
    }
    div[data-testid="stHorizontalBlock"] > div {
        gap: 0.75rem;
    }
</style>""",unsafe_allow_html=True)
STREAMS=['OIL','GAS','WATER','COND']
MODELS=['Seasonal Naive','Linear Trend','Random Forest','XGBoost']
@st.cache_data(show_spinner=False)
def load_data():
    d=pd.read_excel('production.xlsx',sheet_name='UTOROGU')
    d['DATE']=pd.to_datetime(d['DATE'],errors='coerce')
    d=d.dropna(subset=['DATE'])
    d['DATE']=d['DATE'].dt.to_period('M').dt.to_timestamp()
    for c in STREAMS+['DAYS']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d

def aggregate(data):
    # A missing month is missing, not automatically zero production.
    return data.groupby('DATE')[STREAMS].sum(min_count=1).sort_index().asfreq('MS')

def design_matrix(dates,origin):
    t=np.array([(d.year-origin.year)*12+d.month-origin.month for d in dates],dtype=float)
    mo=np.array([d.month for d in dates]);return np.column_stack([t,np.sin(2*np.pi*mo/12),np.cos(2*np.pi*mo/12)])

def predict(history,dates,name):
    if len(dates)==0:return np.array([])
    if name=='Seasonal Naive':
        # Match calendar month, even for a cutoff that is not December.
        last_year=history.iloc[-12:]
        by_month={d.month:v for d,v in last_year.items()}
        return np.array([max(0,float(by_month[d.month])) for d in dates])
    x=design_matrix(history.index,history.index[0]);future=design_matrix(dates,history.index[0])
    if name=='Linear Trend': model=LinearRegression()
    elif name=='Random Forest': model=RandomForestRegressor(n_estimators=130,min_samples_leaf=3,random_state=42,n_jobs=-1)
    elif name=='XGBoost':
        from xgboost import XGBRegressor
        model=XGBRegressor(n_estimators=180,max_depth=3,learning_rate=.04,objective='reg:squarederror',random_state=42,n_jobs=2)
    else:raise ValueError('Unknown model')
    model.fit(x,history.to_numpy());return np.maximum(0,np.asarray(model.predict(future),dtype=float))

def prepare_history(monthly,stream,cutoff):
    s=monthly.loc[:cutoff,stream]
    if len(s)<40:raise ValueError('At least 40 months of training history are required.')
    if s.isna().any():
        # Interpolate only within training history, never across the selected cutoff.
        s=s.interpolate(method='time',limit_direction='both')
    if s.isna().any():raise ValueError('Missing historical observations cannot be resolved.')
    return s

def score(a,p):
    a=np.asarray(a);p=np.asarray(p)
    return {'MAE':mean_absolute_error(a,p),'RMSE':np.sqrt(mean_squared_error(a,p)),
            'sMAPE (%)':100*np.mean(2*np.abs(a-p)/(np.abs(a)+np.abs(p)+1e-9))}

def backtest(monthly,stream,models,horizon,folds):
    dates=monthly.index
    origins=[d for d in dates if d.month==12 and d.year>=1990 and d+pd.DateOffset(months=horizon)<=dates[-1]]
    origins=origins[-folds:]
    rows=[]
    for origin in origins:
        h=prepare_history(monthly,stream,origin)
        future=pd.date_range(origin+pd.offsets.MonthBegin(1),periods=horizon,freq='MS')
        actual=monthly[stream].reindex(future)
        valid=actual.notna().to_numpy()
        if not valid.any():continue
        for name in models:
            try:
                pred=predict(h,future,name)
                metrics=score(actual.to_numpy()[valid],pred[valid])
                rows.append({'Model':name,'Origin':origin.strftime('%Y-%m'),'Horizon (months)':horizon,'Observed months':int(valid.sum()),**metrics})
            except Exception as exc:st.warning(f'{name}, {origin:%Y-%m}: {exc}')
    return pd.DataFrame(rows)

source=load_data()
with st.sidebar:
    st.subheader('00 / Automated data quality')
    thesis_mode=st.toggle('Thesis-aligned historical cutoff (Dec 2014)',value=True,help='The uploaded thesis uses history through December 2014, rather than the raw file endpoint of December 2015.')
    coverage=st.slider('Minimum completion coverage',0,100,40,5,help='Coverage of observed oil months over each completion’s active recorded span')
    negative_action=st.selectbox('Negative production values',['Flag and retain','Convert to missing'])
    duplicates=st.selectbox('Repeated completion-month rows',['Aggregate recorded values','Keep first record'])
raw,quality_report=clean_production(source,duplicate_policy='sum' if duplicates.startswith('Aggregate') else 'first',negative_policy='missing' if negative_action.startswith('Convert') else 'flag',coverage_threshold=coverage/100,end_date='2014-12-01' if thesis_mode else None)

st.caption('OML 34  /  UTOROGU FIELD  /  INTERACTIVE RESEARCH WORKSPACE')
st.title('◈ Utorogu Production Intelligence')
st.write('Historical production analytics · Fixed-origin forecasts · Multi-origin model validation')
with st.sidebar:
    st.subheader('01 / Data scope')
    scope=st.radio('Aggregation',['Entire field','Selected completions'],horizontal=False)
    if scope=='Selected completions':
        well_ids=sorted(raw['UNIQUEID'].dropna().astype(str).unique())
        selected=st.multiselect('Well completion IDs',well_ids,default=well_ids[:1])
        subset=raw[raw['UNIQUEID'].astype(str).isin(selected)]
    else:subset=raw
    st.subheader('02 / Forecast setup')
    stream=st.selectbox('Production stream',STREAMS)
    model=st.selectbox('Forecasting model',MODELS,index=3)
    monthly=aggregate(subset) if not subset.empty else pd.DataFrame()
    if not monthly.empty:
        options=[d for d in monthly.index if d.year>=1990 and d.month==12 and d<=monthly.index[-1]]
        if monthly.index[-1] not in options:options.append(monthly.index[-1])
        cutoff=st.selectbox('Train through',options,index=len(options)-1,format_func=lambda d:d.strftime('%b %Y'))
        end_year=st.slider('Forecast to year',int(cutoff.year)+1,2050,max(2035,int(cutoff.year)+1))
        display_year=st.slider('Chart display from',int(monthly.index[0].year),end_year,max(int(cutoff.year)-8,int(monthly.index[0].year)))
        sensitivity=st.checkbox('±20% scenario sensitivity',True)
        show_actual=st.checkbox('Show post-cutoff observations',True)
        run=st.button('▶  TRAIN & FORECAST',type='primary',use_container_width=True)
    else:run=False
    st.caption('Forecast sensitivity is not a statistical confidence interval. Production units and reporting basis require source verification.')

if subset.empty:
    st.warning('Select at least one completion to analyze.');st.stop()

st.session_state.setdefault('forecast',None)
config=(thesis_mode,coverage,negative_action,duplicates)
if st.session_state.get('cleaning_config')!=config:
    st.session_state['forecast']=None
    st.session_state.pop('backtest_key',None)
    st.session_state['cleaning_config']=config

if run:
    try:
        hist=prepare_history(monthly,stream,cutoff)
        future=pd.date_range(cutoff+pd.offsets.MonthBegin(1),pd.Timestamp(end_year,12,1),freq='MS')
        with st.spinner('Fitting model and generating consecutive monthly forecasts…'):
            p=predict(hist,future,model)
        result=pd.DataFrame({'Date':future,'Forecast':p,'Actual':monthly[stream].reindex(future).to_numpy(),
                             'SensitivityLow':p*.8,'SensitivityHigh':p*1.2})
        result['Stream']=stream;result['Model']=model;result['TrainingCutoff']=cutoff.strftime('%Y-%m')
        result['Scope']='Entire field' if scope=='Entire field' else f'{len(selected)} selected completions'
        st.session_state.forecast=(result,hist,stream,model,cutoff,scope,tuple(selected) if scope!='Entire field' else ())
        st.success(f'Generated {len(result):,} consecutive forecast months for {stream} with {model}.')
    except Exception as exc:st.error(f'Forecast failed: {type(exc).__name__}: {exc}')

v1,v2,v3,v4=st.columns(4)
v1.metric('Records in scope',f'{len(subset):,}')
v2.metric('Completion IDs',f'{subset.UNIQUEID.nunique():,}')
v3.metric('Observed months',f'{monthly[stream].notna().sum():,}')
v4.metric('Historical end',monthly.index[-1].strftime('%b %Y'))

tab_forecast,tab_explore,tab_validate,tab_data,tab_clean=st.tabs(['📈 Forecast studio','◫ Historical analytics','◎ Model backtesting','▤ Data & methodology','🧹 Cleaning studio'])
with tab_forecast:
    item=st.session_state.forecast
    current_scope=tuple(selected) if scope!='Entire field' else ()
    if item and (item[2]!=stream or item[3]!=model or item[5]!=scope or item[6]!=current_scope):
        st.info('The controls have changed. Select TRAIN & FORECAST to calculate a result for the new configuration.')
    elif item:
        result,hist,s,m,c,_,_=item
        overlap=result.dropna(subset=['Actual']);shown=result[result.Date.dt.year>=display_year]
        a,b,d=st.columns(3)
        a.metric('Training months',len(hist));b.metric('Forecast months',len(result))
        d.metric('Fixed-origin overlap MAE',f'{mean_absolute_error(overlap.Actual,overlap.Forecast):,.2f}' if len(overlap) else 'N/A')
        fig=go.Figure()
        h=hist[hist.index.year>=display_year]
        fig.add_trace(go.Scatter(x=h.index,y=h.values,name='Training history',line=dict(color='#93a7c1',width=2)))
        if show_actual and not overlap.empty:
            obs=overlap[overlap.Date.dt.year>=display_year]
            fig.add_trace(go.Scatter(x=obs.Date,y=obs.Actual,name='Observed after origin',line=dict(color='#45dca8',width=2)))
        if sensitivity and not shown.empty:
            fig.add_trace(go.Scatter(x=shown.Date,y=shown.SensitivityHigh,line=dict(width=0),showlegend=False,hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=shown.Date,y=shown.SensitivityLow,line=dict(width=0),fill='tonexty',fillcolor='rgba(21,194,191,.18)',name='±20% scenario',hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=shown.Date,y=shown.Forecast,name=m+' forecast',line=dict(color='#36d8e6',width=3)))
        fig.add_vline(x=c.timestamp()*1000,line_dash='dash',line_color='#f4c565')
        fig.update_layout(template='plotly_dark',paper_bgcolor='#0d2032',plot_bgcolor='#0d2032',height=540,
                          hovermode='x unified',title=f'{s} | {m} | Training origin {c:%Y-%m}',xaxis_title='Month',
                          yaxis_title='Recorded source units (unverified)',legend=dict(orientation='h',y=1.13))
        st.plotly_chart(fig,use_container_width=True)
        if len(overlap):
            metrics=score(overlap.Actual,overlap.Forecast)
            st.caption('Fixed-origin observed overlap: '+', '.join(f'{k}: {v:,.2f}' for k,v in metrics.items())+'; this is not rolling-origin validation.')
        st.dataframe(shown,hide_index=True,use_container_width=True)
        st.download_button('↓ Export all forecast months (CSV)',result.to_csv(index=False).encode(),
                           file_name=f'utorogu_{s.lower()}_{m.lower().replace(" ","_")}_{c:%Y%m}.csv',mime='text/csv')
    else:st.info('Select a model and click TRAIN & FORECAST in the sidebar to generate the first forecast.')
with tab_explore:
    st.subheader('Production history')
    show_streams=st.multiselect('Streams to plot',STREAMS,default=[stream])
    if show_streams:
        fig=go.Figure()
        for name in show_streams:fig.add_trace(go.Scatter(x=monthly.index,y=monthly[name],name=name,mode='lines'))
        fig.update_layout(template='plotly_dark',paper_bgcolor='#0d2032',plot_bgcolor='#0d2032',height=480,
                          hovermode='x unified',xaxis_title='Month',yaxis_title='Recorded source units (unverified)')
        st.plotly_chart(fig,use_container_width=True)
    st.subheader('Yearly recorded totals')
    annual=monthly[STREAMS].resample('YS').sum(min_count=1)
    st.dataframe(annual.assign(Year=annual.index.year).set_index('Year'),use_container_width=True)
    st.download_button('↓ Export aggregated monthly data',monthly.reset_index().to_csv(index=False).encode(),file_name='utorogu_monthly_history.csv')
with tab_validate:
    st.subheader('Rolling-origin, fixed-horizon backtesting')
    st.write('Each fold fits only data available through its December origin and forecasts all subsequent months without using future observations as model inputs.')
    col1,col2,col3=st.columns(3)
    chosen=col1.multiselect('Compare models',MODELS,default=MODELS)
    horizon=col2.selectbox('Forecast horizon (months)',[6,12,24,36],index=1)
    folds=col3.slider('Most recent eligible origins',2,12,5)
    if st.button('Run rolling-origin evaluation',use_container_width=True):
        if not chosen:st.warning('Choose at least one model.')
        else:
            with st.spinner('Evaluating models across historical origins…'):
                report=backtest(monthly,stream,chosen,horizon,folds)
            if report.empty:st.warning('No eligible origins with observed post-cutoff data were found.')
            else:
                st.session_state['backtest_report']=report
                st.session_state['backtest_key']=(scope,current_scope,stream,horizon,folds,tuple(chosen))
    key=(scope,current_scope,stream,horizon,folds,tuple(chosen))
    if st.session_state.get('backtest_key')==key:
        report=st.session_state['backtest_report']
        summary=report.groupby('Model',as_index=False).agg(Folds=('Origin','nunique'),MAE=('MAE','mean'),RMSE=('RMSE','mean'),sMAPE_pct=('sMAPE (%)','mean')).sort_values('RMSE')
        st.dataframe(summary,use_container_width=True,hide_index=True)
        st.bar_chart(summary.set_index('Model')['RMSE'])
        with st.expander('Per-origin validation results'):st.dataframe(report,use_container_width=True,hide_index=True)
        st.download_button('↓ Export backtest results',report.to_csv(index=False).encode(),file_name='utorogu_backtesting.csv')
with tab_data:
    st.subheader('Dataset quality & forecasting assumptions')
    st.write(f'Source sheet: **UTOROGU** · Source records: **{len(raw):,}** · Completion IDs: **{raw.UNIQUEID.nunique():,}** · Date range: **{raw.DATE.min():%Y-%m} to {raw.DATE.max():%Y-%m}**.')
    missing=monthly[STREAMS].isna().sum().rename('Missing monthly values').to_frame()
    st.dataframe(missing,use_container_width=True)
    st.markdown('''**Methods and limitations**
- Aggregation sums monthly recorded values across selected completions. Whether the source numbers are rates or volumes is **not documented**; do not infer units.
- Missing monthly observations remain missing in raw aggregation. Missing training months are interpolated using only the training window.
- All forecasts are fixed-origin multi-step forecasts. Post-origin actuals are used only for evaluation.
- Seasonal Naive repeats the last observed value from the same calendar month; linear trend, Random Forest and XGBoost use elapsed month and annual sine/cosine seasonality.
- ±20% scenarios are sensitivity envelopes, **not** confidence intervals. Long-range forecasts are illustrative and should not be treated as reserves estimates.
- Backtest rankings depend on chosen horizon, origins and available observed data; they are not a substitute for reservoir engineering validation.''')
    st.dataframe(subset.head(100),use_container_width=True,hide_index=True)

with tab_clean:
    st.subheader('Automated data cleaning & audit trail')
    st.write('Every transformation is recorded. Source values are preserved in the original Excel file. Missing records are not assumed to represent zero production.')
    a,b,c=st.columns(3)
    a.metric('Original rows',f'{len(source):,}')
    b.metric('Cleaned completion-months',f'{len(raw):,}')
    c.metric('Retained completions',f'{raw.UNIQUEID.nunique():,}')
    st.dataframe(quality_report,use_container_width=True,hide_index=True)
    st.download_button('↓ Download cleaning audit CSV',quality_report.to_csv(index=False).encode(),file_name='utorogu_cleaning_audit.csv',mime='text/csv')
    st.download_button('↓ Download cleaned entity-level CSV',raw.to_csv(index=False).encode(),file_name='utorogu_cleaned_entity_monthly.csv',mime='text/csv')
    st.markdown('**Missingness by stream**')
    st.dataframe(raw[STREAMS].isna().sum().rename('Missing values').to_frame(),use_container_width=True)
    st.caption('Outliers are flagged, not clipped. Any subsequent model imputation is restricted to its own training window. Coverage filtering changes the modeled field population and must be disclosed in research reports.')
