import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm, chi2
from jinja2 import Template
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ===================== Page Configuration =====================
st.set_page_config(page_title="RAISE_X | Market Risk Analysis", layout="wide")

st.markdown("""
<style>
    .main-header { text-align: center; color: #2E8B57; font-size: 2.2rem; font-weight: bold; }
    .report-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #2E8B57; margin-bottom: 20px; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ===================== Core Math Functions =====================
def to_pct(log_val):
    return (np.exp(log_val) - 1) * 100

def kupiec_test(breaches, total, p_expected):
    if total == 0 or breaches == 0: return 0, 1.0
    p_obs = breaches / total
    # Likelihood Ratio for Proportion of Failures
    num = ((1 - p_expected)**(total - breaches)) * (p_expected**breaches)
    den = ((1 - p_obs)**(total - breaches)) * (p_obs**breaches)
    if den == 0: den = 1e-10
    lr = -2 * np.log(num / den)
    p_value = 1 - chi2.cdf(lr, df=1)
    return lr, p_value

# ===================== Sidebar =====================
with st.sidebar:
    st.title("⚙️ Analysis Control")
    conf_level = st.slider("Confidence Interval (%)", 90, 99, 95) / 100.0
    num_sims = st.number_input("Simulations", 1000, 50000, 10000)
    window = st.number_input("Rolling Window (Weeks)", 20, 104, 52)
    
    st.divider()
    st.markdown("### Data Pre-processing")
    treat_outliers = st.toggle("Remove Price Outliers (IQR Method)", value=True)
    
    st.info("RAISE_X: Real-time Insights on Scenario Evaluation")

# ===================== App Header =====================
st.markdown('<h1 class="main-header">RAISE_X: Market Risk & Scenario Evaluation</h1>', unsafe_allow_html=True)

# ===================== File Loading & Cleaning =====================
uploaded_file = st.file_uploader("Upload Market Data (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        # Load
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Date Handling
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
        
        # Column Selection
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        p_col = st.selectbox("Market Price (Modal)", cols, index=cols.index('Modal') if 'Modal' in cols else 0)
        a_col = st.selectbox("Arrivals (Optional)", ["None"] + cols, index=cols.index('Arrivals')+1 if 'Arrivals' in cols else 0)

        # Cleaning Logic
        working_df = df[[p_col]].copy()
        if a_col != "None": working_df[a_col] = df[a_col]
        
        # Outlier Treatment
        if treat_outliers:
            Q1 = working_df[p_col].quantile(0.25)
            Q3 = working_df[p_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            working_df = working_df[(working_df[p_col] >= lower_bound) & (working_df[p_col] <= upper_bound)]
            st.caption(f"Outlier treatment removed {len(df) - len(working_df)} extreme weeks.")

        # Returns
        working_df['Log_Returns'] = np.log(working_df[p_col] / working_df[p_col].shift(1))
        if a_col != "None":
            working_df['Log_Arrivals'] = np.log(working_df[a_col].replace(0, np.nan)).fillna(method='bfill')
        working_df.dropna(inplace=True)

        # ===================== Tabs =====================
        t1, t2, t3 = st.tabs(["📊 Static Analysis", "🧪 Backtesting (Kupiec)", "🔮 Forecasting"])

        with t1:
            st.subheader("Scientifically Valid Risk Metrics")
            mu, sigma = working_df['Log_Returns'].mean(), working_df['Log_Returns'].std()
            z_score = norm.ppf(1 - conf_level)

            # Calculations
            h_var = np.percentile(working_df['Log_Returns'], (1 - conf_level) * 100)
            p_var = mu + z_score * sigma
            
            if a_col != "None":
                X = sm.add_constant(working_df['Log_Arrivals'])
                res = sm.OLS(working_df['Log_Returns'], X).fit()
                sim_arr = np.random.normal(working_df['Log_Arrivals'].mean(), working_df['Log_Arrivals'].std(), num_sims)
                sim_mu = res.params[0] + res.params[1] * sim_arr
                sim_ret = np.random.normal(sim_mu, res.resid.std(), num_sims)
            else:
                sim_ret = np.random.normal(mu, sigma, num_sims)
            
            mc_var = np.percentile(sim_ret, (1 - conf_level) * 100)
            cvar = sim_ret[sim_ret <= mc_var].mean()

            # Display
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Historical VaR", f"{to_pct(h_var):.2f}%")
            m2.metric("Parametric VaR", f"{to_pct(p_var):.2f}%")
            m3.metric("Monte Carlo VaR", f"{to_pct(mc_var):.2f}%")
            m4.metric("Exp. Shortfall (CVaR)", f"{to_pct(cvar):.2f}%")

            # Table
            st.table(pd.DataFrame({
                "Method": ["Historical", "Parametric", "Monte Carlo", "Expected Shortfall (CVaR)"],
                "1-Week Potential Loss (%)": [to_pct(h_var), to_pct(p_var), to_pct(mc_var), to_pct(cvar)]
            }).round(3))

        with t2:
            st.subheader("Model Validation")
            if len(working_df) > window:
                bt_list = []
                for i in range(window, len(working_df)):
                    train = working_df.iloc[i-window:i]
                    act = working_df['Log_Returns'].iloc[i]
                    # Sim for rolling VaR
                    if a_col != "None":
                        m_bt = sm.OLS(train['Log_Returns'], sm.add_constant(train['Log_Arrivals'])).fit()
                        s_r = np.random.normal(m_bt.params[0] + m_bt.params[1] * train['Log_Arrivals'].mean(), m_bt.resid.std(), 1000)
                    else:
                        s_r = np.random.normal(train['Log_Returns'].mean(), train['Log_Returns'].std(), 1000)
                    
                    v_lim = np.percentile(s_r, (1 - conf_level) * 100)
                    bt_list.append({'Actual': act, 'VaR': v_lim, 'Breach': act < v_lim})
                
                bt_df = pd.DataFrame(bt_list)
                total_b = bt_df['Breach'].sum()
                lr, p_val = kupiec_test(total_b, len(bt_df), 1 - conf_level)
                
                c1, c2 = st.columns(2)
                c1.metric("Breaches Found", total_b)
                c2.metric("Kupiec P-Value", f"{p_val:.4f}")
                
                if p_val < 0.05: st.error("Model under-estimated tail risk (Rejected).")
                else: st.success("Model captures volatility accurately (Accepted).")
            else:
                st.warning("Insufficient history for rolling backtest.")

        with t3:
            st.subheader("Supply-Adjusted Forecast")
            if a_col != "None":
                model_final = sm.OLS(working_df['Log_Returns'], sm.add_constant(working_df['Log_Arrivals'])).fit()
                f_data = []
                for w in range(1, 5):
                    # Simulate arrivals based on current trend
                    s_arrivals = np.random.normal(working_df['Log_Arrivals'].iloc[-12:].mean(), working_df['Log_Arrivals'].std(), 5000)
                    s_mu = model_final.params[0] + model_final.params[1] * s_arrivals
                    s_returns = np.random.normal(s_mu, model_final.resid.std(), 5000)
                    f_data.append({"Week": w, "VaR Loss (%)": to_pct(np.percentile(s_returns, (1 - conf_level) * 100))})
                st.dataframe(pd.DataFrame(f_data).round(2), use_container_width=True)
            else:
                st.info("Upload Arrivals data for supply-driven forecasts.")

    except Exception as e:
        st.error(f"Computation Error: {e}")
else:
    st.markdown("""
    ### 📂 Ready to Analyze
    Please upload your **Tomato Market Data** to begin the automated VaR analysis.
    The app will automatically calculate:
    * **Downside Risk** across 4 methodologies.
    * **Kupiec POF test** for PhD-level validation.
    * **CVaR (Expected Shortfall)** for tail-risk analysis.
    """)
