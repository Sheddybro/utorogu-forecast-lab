import io
from pathlib import Path
from contextlib import nullcontext
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from data_quality import clean_production
from models import MODELS,forecast,metrics,clean_history

st.set_page_config(page_title='Utorogu | Production Intelligence',page_icon='🛢️',layout='wide',initial_sidebar_state='collapsed')
st.markdown("""<style>
:root { --ink:#252451; --blue:#3658b8; --purple:#7650b5; }
.stApp {background:#f6f7fc;color:#252451}
.block-container {max-width:1540px;padding-top:1.25rem;padding-bottom:3rem;padding-left:2.3rem;padding-right:2.3rem}
[data-testid="stHeader"]{background:rgba(246,247,252,.96)}
section[data-testid="stSidebar"] {background:#202344;min-width:250px!important;max-width:290px!important}
section[data-testid="stSidebar"] > div {width:100%!important}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3 {color:#f1f0ff!important}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#c2c5e2!important}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color:#e2e4f5!important}
[data-testid="stMetric"] {background:white;border:1px solid #e1e3f1;border-radius:13px;padding:16px 18px;min-width:0;box-shadow:0 3px 12px rgba(38,42,91,.045)}
[data-testid="stMetricLabel"] p {font-size:.85rem!important;color:#656987!important;white-space:normal!important}
[data-testid="stMetricValue"] {font-size:clamp(1.3rem,2vw,2rem)!important;color:#292c5a!important;overflow:visible!important;white-space:normal!important;overflow-wrap:anywhere!important}
[data-testid="stMetricDelta"]{display:none}
h1,h2,h3 {color:#252451!important;letter-spacing:-.025em}
.hero {background:linear-gradient(105deg,#303b83 0%,#6946a6 100%);border-radius:16px;padding:24px 30px;margin:0 0 20px;color:#fff}
.hero h1{font-size:clamp(1.6rem,2.3vw,2.35rem)!important;color:#fff!important;margin:0!important;line-height:1.22}
.hero p{color:#e3e3fb;font-size:.91rem;margin:9px 0 0}
.stButton>button[kind="primary"] {background:#5e4aa5;border:0;border-radius:8px}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#e1e3f1}
[data-testid="stPlotlyChart"] {background:#fff;border-radius:12px;border:1px solid #e3e5f1;padding:6px}
[data-testid="stRadio"] div[role="radiogroup"] {gap:.5rem}
@media(max-width:850px){.block-container{padding-left:1rem;padding-right:1rem;padding-top:.7rem}.hero{padding:18px}.hero h1{font-size:1.5rem!important}[data-testid="stMetric"]{padding:11px}}
@media(max-width:640px){
    [data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:.65rem!important}
    [data-testid="stHorizontalBlock"]>[data-testid="column"]{flex:1 1 calc(50% - .65rem)!important;width:calc(50% - .65rem)!important;min-width:calc(50% - .65rem)!important}
    [data-testid="stRadio"] div[role="radiogroup"]{flex-wrap:wrap!important;row-gap:.25rem}
    [data-testid="stRadio"] div[role="radiogroup"] label{white-space:normal!important}
    [data-testid="stPlotlyChart"]{max-width:100%;overflow:hidden}
    [data-testid="stDataFrame"]{max-width:100%;overflow-x:auto}
    .hero{border-radius:12px;padding:16px}
    .hero p{line-height:1.45}
}
</style>""",unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load():
    d=pd.read_excel(Path(__file__).resolve().parent/'production.xlsx',sheet_name='UTOROGU')
    return d

def monthly_sum(d):
    return d.groupby('DATE')[['OIL','GAS','WATER','COND']].sum(min_count=1).sort_index().asfreq('MS')

def plot_series(series,cutoff=None,projection=None,actuals=True,scenario=True):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=series.index,y=series.values,name='Observed production',line=dict(color='#3658b8',width=2.5),mode='lines'))
    if projection is not None:
        f=projection
        if scenario:
            fig.add_trace(go.Scatter(x=list(f.Date)+list(f.Date[::-1]),y=list(f.High)+list(f.Low[::-1]),fill='toself',fillcolor='rgba(118,80,181,.16)',line=dict(color='rgba(0,0,0,0)'),name='±20% sensitivity'))
        fig.add_trace(go.Scatter(x=f.Date,y=f.Forecast,name='Model forecast',line=dict(color='#7650b5',width=3)))
        if actuals and 'Actual' in f:fig.add_trace(go.Scatter(x=f.Date,y=f.Actual,name='Post-cutoff actual',line=dict(color='#566b9a',dash='dot')))
    if cutoff is not None:fig.add_vline(x=cutoff.timestamp()*1000,line_dash='dash',line_color='#8589aa')
    fig.update_layout(height=470,margin=dict(l=22,r=14,t=44,b=25),paper_bgcolor='white',plot_bgcolor='white',font=dict(color='#303354'),hovermode='x unified',legend=dict(orientation='h',y=1.08),xaxis=dict(showgrid=True,gridcolor='#eceefa'),yaxis=dict(showgrid=True,gridcolor='#eceefa',tickformat='~s',automargin=True))
    return fig

raw=load()
with st.sidebar:
    st.markdown('## 🛢️ UTOROGU LAB')
    st.caption('OML 34 · Research-grade forecasting workspace')
    with st.expander('01 · Data quality',expanded=False):
        study_cutoff=st.toggle('Thesis history through Dec 2014',value=True)
        method=st.selectbox('Entity selection method',['Notebook: positive oil-month share','Observed-month coverage','No entity filter'])
        threshold=st.slider('Minimum qualifying share (%)',0,100,40,5)
        negatives=st.selectbox('Negative values',['Flag and retain','Convert to missing'])
        dup=st.selectbox('Duplicate completion-months',['Aggregate by sum','Keep first'])
    st.markdown('### Forecast configuration')
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
    run=st.button('▶ Run forecast',type='primary',width='stretch')
    st.caption('Long-range forecasts are scenarios, not reserve estimates or calibrated confidence intervals.')

st.markdown('<div class="hero"><h1>Utorogu Production Intelligence</h1><p>OML 34  /  Research forecasting workspace  /  Auditable cleaning  /  Chronological validation</p></div>',unsafe_allow_html=True)
cols=st.columns(4,gap='small')
for c,(label,val) in zip(cols,[('Records',f'{len(view):,}'),('Completions',f'{view.UNIQUEID.nunique():,}'),('Monthly periods',f'{monthly[stream].notna().sum():,}'),('Last observation',monthly.index[-1].strftime('%b %Y'))]):c.metric(label,val)

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

st.write('')
section=st.radio('Workspace', ['Forecast studio','Production analytics','Model comparison','Cleaning & audit','Methodology'],horizontal=True,label_visibility='collapsed',key='workspace_nav')
forecast_tab=st.container() if section=='Forecast studio' else nullcontext()
history_tab=st.container() if section=='Production analytics' else st.empty()
compare_tab=st.container() if section=='Model comparison' else st.empty()
clean_tab=st.container() if section=='Cleaning & audit' else st.empty()
method_tab=st.container() if section=='Methodology' else st.empty()

if section=='Forecast studio':
    st.subheader('Forecast and observed production')
    current=st.session_state.get('forecast')
    f=current[1] if current is not None and current[0]==key else None
    if f is None:st.info('Historical production is shown below. Configure a model in the sidebar and select Run forecast to add its projection.')
    series=monthly[stream].loc[:cutoff] if f is not None else monthly[stream]
    st.plotly_chart(plot_series(series,cutoff if f is not None else None,f,show_actual,sensitivity),width='stretch',config={'displaylogo':False})
    if f is not None:
        a,b,c=st.columns(3)
        a.metric('Forecast end',f.Date.iloc[-1].strftime('%b %Y'))
        b.metric('Final projected monthly production',f'{f.Forecast.iloc[-1]:,.1f}')
        c.metric('Forecast months',len(f))
        st.download_button('⬇ Download forecast CSV',f.to_csv(index=False).encode(),'utorogu_forecast.csv','text/csv')
        st.dataframe(f.tail(18),width='stretch',hide_index=True)
if section=='Production analytics':
    st.subheader('Historical field performance')
    st.plotly_chart(plot_series(monthly[stream]),width='stretch')
    annual=monthly[['OIL','GAS','WATER']].resample('YS').sum(min_count=1)
    colors={'OIL':'#3658b8','GAS':'#7650b5','WATER':'#7787b8'}
    st.plotly_chart(go.Figure([go.Bar(x=annual.index.year,y=annual[x],name=x,marker_color=colors[x]) for x in ['OIL','GAS','WATER']]).update_layout(barmode='group',height=380,plot_bgcolor='white',paper_bgcolor='white',font_color='#303354'),width='stretch')
    st.download_button('⬇ Download monthly production',monthly.to_csv().encode(),'utorogu_monthly_history.csv','text/csv')
if section=='Model comparison':
    st.subheader('Chronological, fixed-origin model comparison')
    st.caption('Thesis retrospective window: 2006–2014. December-origin rolling tests; the chosen horizon must end by December 2014. Each model trains only on data available at its origin.')
    a,b,c=st.columns([2,1,1])
    chosen=a.multiselect('Models to compare',MODELS,default=['Seasonal Naive','Arps Exponential DCA','XGBoost'])
    horizon=b.selectbox('Forecast horizon (months)',[12,24,36],index=1)
    folds=c.slider('Historical origins',1,5,3)
    if st.button('Run chronological benchmark',type='primary'):
        dates=[dt for dt in monthly.index if dt.month==12 and dt.year>=2005 and dt.year<=2013 and dt+pd.DateOffset(months=horizon)<=pd.Timestamp('2014-12-01')][-folds:]
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
        st.dataframe(rank,width='stretch',hide_index=True)
        st.bar_chart(rank.set_index('Model')['RMSE'])
        st.download_button('⬇ Download validation results',detail.to_csv(index=False).encode(),'utorogu_model_comparison.csv','text/csv')
        with st.expander('All validation folds'):st.dataframe(detail,width='stretch',hide_index=True)
if section=='Cleaning & audit':
    st.subheader('Automated cleaning report')
    st.info('The original workbook remains unchanged. Negative and duplicate policies are configurable. Outliers are flagged, not silently removed.')
    st.dataframe(report,width='stretch',hide_index=True)
    st.markdown('#### Completion selection review')
    summary=d.groupby('UNIQUEID').agg(First=('DATE','min'),Last=('DATE','max'),Months=('DATE','nunique'),Positive_oil_months=('OIL',lambda x:int((x>0).sum()))).reset_index()
    st.dataframe(summary,width='stretch',hide_index=True)
    st.download_button('⬇ Download cleaned completion records',d.to_csv(index=False).encode(),'utorogu_cleaned_completions.csv','text/csv')
    st.download_button('⬇ Download cleaning audit',report.to_csv(index=False).encode(),'utorogu_cleaning_audit.csv','text/csv')
if section=='Methodology':
    st.subheader('Research alignment and interpretation')
    st.markdown('''**Source:** Utorogu OML 34 completion-month production workbook. The accompanying thesis and notebook identify a research modelling endpoint of **December 2014**, a **2006–2014** retrospective evaluation period, and a **2027–2035** long-range projection period.\n\n**Models:** Seasonal Naive, Arps exponential decline, Prophet, XGBoost, LSTM, Prophet–XGBoost residual hybrid, equal-weight and internal-validation-weighted ensembles. Linear Trend and Random Forest are additional diagnostic baselines.\n\n**Feature and validation safety:** Forecasts are generated from observations through the selected origin only. XGBoost uses calendar/time features for recursive-free horizon extrapolation; this is a deliberately simplified implementation and **not a bit-for-bit replication of the notebook's lagged multi-stream XGBoost model**. LSTM uses a univariate recursive 12-month sequence. Ensemble weights are estimated on an internal 24-month calibration window, not on the external evaluation period.\n\n**Data limitations:** Missing entity-month observations are not assumed to be zero. Reporting units are not verified. ±20% bands are illustrative sensitivities, not statistical confidence intervals. Condensate is exploratory because the thesis identifies sparse coverage. Historical-only models cannot anticipate interventions or operational changes.\n\n**Research reproducibility:** Record your cleaning policy, retained completions, train/test dates, stream, model, horizon, and software versions alongside exported results. The notebook's exact 47-entity population must be verified before claiming direct thesis replication.''')
