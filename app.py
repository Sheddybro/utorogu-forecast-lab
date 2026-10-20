import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from data_quality import clean_production
from models import MODELS,forecast,metrics,clean_history

st.set_page_config(page_title='Utorogu | Forecast Research Studio',page_icon='🛢️',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp {background:#f4f7fb;color:#172b46}
[data-testid="stHeader"]{background:#f4f7fb}
section[data-testid="stSidebar"]{background:#10263d}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{color:#f4f8ff!important}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{color:#dce9f7!important}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#b9cfe3!important}
[data-testid="stMetric"]{background:#fff;border:1px solid #e2eaf3;border-radius:14px;padding:16px 20px;box-shadow:0 5px 18px rgba(22,49,79,.04)}
[data-testid="stMetricLabel"] p{color:#526781!important}
[data-testid="stMetricValue"]{color:#102e4d!important}
.stTabs [data-baseweb="tab-list"]{gap:8px;flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{background:#e8eff7;border-radius:9px 9px 0 0;padding:10px 15px;color:#183451}
.stTabs [aria-selected="true"]{background:#fff;color:#067e86;border-bottom:3px solid #13a3a7}
h1,h2,h3{color:#142f4c!important}
.hero{background:linear-gradient(115deg,#102d48,#175b6a);padding:28px 32px;border-radius:18px;color:white;margin-bottom:18px}
.hero h1{color:white!important;margin:0;font-size:2.2rem}
.hero p{color:#d4e8f0;margin:8px 0 0}
.stButton>button[kind="primary"]{background:#087f87;border:0;border-radius:9px}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load():
    d=pd.read_excel('production.xlsx',sheet_name='UTOROGU')
    return d

def monthly_sum(d):
    return d.groupby('DATE')[['OIL','GAS','WATER','COND']].sum(min_count=1).sort_index().asfreq('MS')

def plot_series(series,cutoff=None,projection=None,actuals=True,scenario=True):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=series.index,y=series.values,name='Observed production',line=dict(color='#087e92',width=2.5),mode='lines'))
    if projection is not None:
        f=projection
        if scenario:
            fig.add_trace(go.Scatter(x=list(f.Date)+list(f.Date[::-1]),y=list(f.High)+list(f.Low[::-1]),fill='toself',fillcolor='rgba(236,154,67,.18)',line=dict(color='rgba(0,0,0,0)'),name='±20% sensitivity'))
        fig.add_trace(go.Scatter(x=f.Date,y=f.Forecast,name='Model forecast',line=dict(color='#e18b32',width=3)))
        if actuals and 'Actual' in f:fig.add_trace(go.Scatter(x=f.Date,y=f.Actual,name='Post-cutoff actual',line=dict(color='#52618c',dash='dot')))
    if cutoff is not None:fig.add_vline(x=cutoff.timestamp()*1000,line_dash='dash',line_color='#8495a8')
    fig.update_layout(height=450,margin=dict(l=8,r=8,t=20,b=15),paper_bgcolor='white',plot_bgcolor='white',font=dict(color='#203e59'),hovermode='x unified',legend=dict(orientation='h',y=1.08),xaxis=dict(showgrid=True,gridcolor='#edf2f7'),yaxis=dict(showgrid=True,gridcolor='#edf2f7',title='Reported production (units unverified)'))
    return fig

raw=load()
with st.sidebar:
    st.markdown('## 🛢️ UTOROGU LAB')
    st.caption('OML 34 · Research-grade forecasting workspace')
    st.markdown('### 01 · Data quality')
    study_cutoff=st.toggle('Thesis history through Dec 2014',value=True)
    method=st.selectbox('Entity selection method',['Notebook: positive oil-month share','Observed-month coverage','No entity filter'])
    threshold=st.slider('Minimum qualifying share (%)',0,100,40,5)
    negatives=st.selectbox('Negative values',['Flag and retain','Convert to missing'])
    dup=st.selectbox('Duplicate completion-months',['Aggregate by sum','Keep first'])
    st.markdown('### 02 · Forecast configuration')
    stream=st.selectbox('Production stream',['OIL','GAS','WATER','COND'])
    model=st.selectbox('Forecast model',MODELS,index=5)

# Clean with transparent rules, then apply notebook selection if requested.
d,report=clean_production(raw,duplicate_policy='sum' if dup.startswith('Aggregate') else 'first',negative_policy='missing' if negatives.startswith('Convert') else 'flag',coverage_threshold=threshold/100 if method=='Observed-month coverage' else 0,end_date='2014-12-01' if study_cutoff else None)
if method=='Notebook: positive oil-month share':
    group=d.groupby('UNIQUEID')['OIL']
    shares=group.apply(lambda s:float((s.fillna(0)>0).mean()))
    keep=shares[shares>=threshold/100].index
    excluded=int(d.UNIQUEID.nunique()-len(keep))
    d=d[d.UNIQUEID.isin(keep)].copy()
    report=pd.concat([report,pd.DataFrame([{'Check':'Notebook positive-oil-month selection','Affected':excluded,'Action':f'Excluded completions with positive OIL share below {threshold}%'}])],ignore_index=True)
with st.sidebar:
    scope=st.radio('Analysis scope',['Field total','Selected completions'])
    if scope=='Selected completions':
        ids=sorted(d.UNIQUEID.astype(str).unique())
        selected=st.multiselect('Completion identifiers',ids,default=ids[:1])
        view=d[d.UNIQUEID.astype(str).isin(selected)].copy()
    else:view=d.copy()
    if view.empty:st.error('No completions selected or retained.');st.stop()
    monthly=monthly_sum(view)
    opts=[dt for dt in monthly.index if dt.month==12 and dt.year>=1990]
    if not opts:st.error('Not enough dated history for forecasting.');st.stop()
    cutoff=st.selectbox('Training end',opts,index=len(opts)-1,format_func=lambda x:x.strftime('%b %Y'))
    end_year=st.slider('Forecast end year',cutoff.year+1,2050,max(2035,cutoff.year+1))
    show_actual=st.toggle('Show post-cutoff observations',value=True)
    sensitivity=st.toggle('Show ±20% sensitivity',value=True)
    run=st.button('▶ Run forecast',type='primary',use_container_width=True)
    st.caption('Long-range forecasts are scenarios, not reserve estimates or calibrated confidence intervals.')

st.markdown('<div class="hero"><h1>Utorogu Production Intelligence</h1><p>Research dashboard · Automated data quality · 10 forecasting approaches · Chronological evaluation</p></div>',unsafe_allow_html=True)
cols=st.columns(4)
for c,(label,val) in zip(cols,[('Retained records',f'{len(view):,}'),('Completions',f'{view.UNIQUEID.nunique():,}'),('Observed months',f'{monthly[stream].notna().sum():,}'),('Historical endpoint',monthly.index[-1].strftime('%b %Y'))]):c.metric(label,val)

# Do not display results produced using different settings.
key=(stream,model,str(cutoff),end_year,scope,tuple(selected) if scope!='Field total' else (),study_cutoff,method,threshold,negatives,dup)
if run:
    try:
        h=clean_history(monthly.loc[:cutoff,stream]);dates=pd.date_range(cutoff+pd.offsets.MonthBegin(1),pd.Timestamp(end_year,12,1),freq='MS')
        with st.spinner(f'Training {model}… LSTM and ensembles may take several minutes.'):
            p=forecast(h,dates,model)
        f=pd.DataFrame({'Date':dates,'Forecast':p,'Low':p*.8,'High':p*1.2,'Actual':monthly[stream].reindex(dates).values})
        st.session_state['forecast']=(key,f)
        st.success(f'{model}: {len(f)} monthly forecasts generated.')
    except Exception as exc:st.error(f'{type(exc).__name__}: {exc}')

forecast_tab,history_tab,compare_tab,clean_tab,method_tab=st.tabs(['📈 Forecast studio','📊 Production analytics','🏆 Model comparison','🧹 Cleaning & audit','📖 Methodology'])
with forecast_tab:
    st.subheader('Forecast and observed production')
    current=st.session_state.get('forecast')
    f=current[1] if current is not None and current[0]==key else None
    if f is None:st.info('Historical production is shown below. Configure a model in the sidebar and select Run forecast to add its projection.')
    series=monthly[stream].loc[:cutoff] if f is not None else monthly[stream]
    st.plotly_chart(plot_series(series,cutoff if f is not None else None,f,show_actual,sensitivity),use_container_width=True)
    if f is not None:
        a,b,c=st.columns(3)
        a.metric('Forecast end',f.Date.iloc[-1].strftime('%b %Y'))
        b.metric('Final projected monthly production',f'{f.Forecast.iloc[-1]:,.1f}')
        c.metric('Forecast months',len(f))
        st.download_button('⬇ Download forecast CSV',f.to_csv(index=False).encode(),'utorogu_forecast.csv','text/csv')
        st.dataframe(f.tail(18),use_container_width=True,hide_index=True)
with history_tab:
    st.subheader('Historical field performance')
    st.plotly_chart(plot_series(monthly[stream]),use_container_width=True)
    annual=monthly[['OIL','GAS','WATER']].resample('YS').sum(min_count=1)
    st.plotly_chart(go.Figure([go.Bar(x=annual.index.year,y=annual[x],name=x) for x in ['OIL','GAS','WATER']]).update_layout(barmode='group',height=380,plot_bgcolor='white',paper_bgcolor='white',font_color='#203e59'),use_container_width=True)
    st.download_button('⬇ Download monthly production',monthly.to_csv().encode(),'utorogu_monthly_history.csv','text/csv')
with compare_tab:
    st.subheader('Chronological, fixed-origin model comparison')
    st.caption('Each model trains using only observations available at the origin. Future actuals are used for scoring, not prediction. Validation-weighted ensemble calibration occurs inside its training history.')
    a,b,c=st.columns([2,1,1])
    chosen=a.multiselect('Models to compare',MODELS,default=['Seasonal Naive','Arps Exponential DCA','XGBoost'])
    horizon=b.selectbox('Forecast horizon (months)',[12,24,36],index=1)
    folds=c.slider('Historical origins',1,5,3)
    if st.button('Run chronological benchmark',type='primary'):
        dates=[dt for dt in monthly.index if dt.month==12 and dt.year>=1990 and dt+pd.DateOffset(months=horizon)<=monthly.index[-1]][-folds:]
        rows=[]
        with st.spinner('Comparing models…'):
            for origin in dates:
                try:h=clean_history(monthly.loc[:origin,stream])
                except Exception:continue
                future=pd.date_range(origin+pd.offsets.MonthBegin(1),periods=horizon,freq='MS')
                actual=monthly[stream].reindex(future).values
                for m in chosen:
                    try:
                        pred=forecast(h,future,m)
                        rows.append({'Model':m,'Origin':origin.strftime('%Y-%m'),**metrics(actual,pred)})
                    except Exception as e:st.warning(f'{m} at {origin:%Y-%m}: {e}')
        st.session_state['benchmark']=(key,pd.DataFrame(rows))
    result=st.session_state.get('benchmark')
    if result and result[0]==key and not result[1].empty:
        detail=result[1];rank=detail.groupby('Model',as_index=False)[['MAE','RMSE','sMAPE (%)']].mean().sort_values('RMSE')
        st.dataframe(rank,use_container_width=True,hide_index=True)
        st.bar_chart(rank.set_index('Model')['RMSE'])
        st.download_button('⬇ Download validation results',detail.to_csv(index=False).encode(),'utorogu_model_comparison.csv','text/csv')
        with st.expander('All validation folds'):st.dataframe(detail,use_container_width=True,hide_index=True)
with clean_tab:
    st.subheader('Automated cleaning report')
    st.info('The original workbook remains unchanged. Negative and duplicate policies are configurable. Outliers are flagged, not silently removed.')
    st.dataframe(report,use_container_width=True,hide_index=True)
    st.markdown('#### Completion selection review')
    summary=d.groupby('UNIQUEID').agg(First=('DATE','min'),Last=('DATE','max'),Months=('DATE','nunique'),Positive_oil_months=('OIL',lambda x:int((x>0).sum()))).reset_index()
    st.dataframe(summary,use_container_width=True,hide_index=True)
    st.download_button('⬇ Download cleaned completion records',d.to_csv(index=False).encode(),'utorogu_cleaned_completions.csv','text/csv')
    st.download_button('⬇ Download cleaning audit',report.to_csv(index=False).encode(),'utorogu_cleaning_audit.csv','text/csv')
with method_tab:
    st.subheader('Research alignment and interpretation')
    st.markdown('''**Source:** Utorogu OML 34 completion-month production workbook. The accompanying thesis and notebook identify a research modelling endpoint of **December 2014**, a **2006–2014** retrospective evaluation period, and a **2027–2035** long-range projection period.\n\n**Models:** Seasonal Naive, Arps exponential decline, Prophet, XGBoost, LSTM, Prophet–XGBoost residual hybrid, equal-weight and internal-validation-weighted ensembles. Linear Trend and Random Forest are additional diagnostic baselines.\n\n**Feature and validation safety:** Forecasts are generated from observations through the selected origin only. XGBoost uses calendar/time features for recursive-free horizon extrapolation; this is a deliberately simplified implementation and **not a bit-for-bit replication of the notebook's lagged multi-stream XGBoost model**. LSTM uses a univariate recursive 12-month sequence. Ensemble weights are estimated on an internal 24-month calibration window, not on the external evaluation period.\n\n**Data limitations:** Missing entity-month observations are not assumed to be zero. Reporting units are not verified. ±20% bands are illustrative sensitivities, not statistical confidence intervals. Condensate is exploratory because the thesis identifies sparse coverage. Historical-only models cannot anticipate interventions or operational changes.\n\n**Research reproducibility:** Record your cleaning policy, retained completions, train/test dates, stream, model, horizon, and software versions alongside exported results. The notebook's exact 47-entity population must be verified before claiming direct thesis replication.''')
