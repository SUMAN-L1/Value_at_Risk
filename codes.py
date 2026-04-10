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
st.set_page_config(page_title="RAISE_X | Market Risk", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { text-align: center; color: #2E8B57; font-size: 2.2rem; font-weight: bold; }
    .report-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #2E8B57; margin-bottom: 20px; }
    .metric-card { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); text-align: center; }
</style>
""", unsafe_allow_html=True)

# ===================== Helper Functions =====================
def to_pct(log_val):
    return (np.exp(log_val) - 1) * 100

def kupiec_test(breaches, total, p_expected):
    if total == 0 or breaches == 0: return 0, 1.0
    p_obs = breaches / total
    # Likelihood Ratio test
    num = ((1 - p_expected)**(total - breaches)) * (p_expected**breaches)
    den = ((1 - p_obs)**(total - breaches)) * (p_obs**breaches)
    if den == 0: den = 1e-10
    lr = -2 * np.log(num / den)
    p_value = 1 - chi2.cdf(lr, df=1)
    return lr, p_value

# ===================== Sidebar =====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/tomato.png", width=80)
    st.title("RAISE_X Settings")
    
    conf_level = st.slider("Confidence Level (%)", 90, 99, 95) / 100.0
    num_sims = st.number_input("Monte Carlo Simulations", 1000, 50000, 10000)
    window = st.number_input("Rolling Window (Weeks)", 20, 104, 52)
    
    st.divider()
    st.info("Developed for Agricultural Economics Research (PhD Thesis Support)")

# ===================== App Header =====================
st.markdown('<h1 class="main-header">RAISE_X: Real-time Insights on Scenario Evaluation</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Automated Market Risk & Supply-Sensitive Forecasting</p>', unsafe_allow_html=True)

# ===================== Data Loading =====================
uploaded_file = st.file_uploader("Upload Market Data (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Standardize Date
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
        
        st.success(f"Successfully loaded {len(df)} weeks of data.")
        
        # Column Selection
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        price_col = st.selectbox("Select Modal Price Column", cols, index=cols.index('Modal') if 'Modal' in cols else 0)
        arrival_col = st.selectbox("Select Arrivals Column (Optional for MC)", ["None"] + cols, index=cols.index('Arrivals')+1 if 'Arrivals' in cols else 0)

        # Pre-processing
        data = df[[price_col]].copy()
        if arrival_col != "None":
            data[arrival_col] = df[arrival_col]
        
        data['Log_Returns'] = np.log(data[price_col] / data[price_col].shift(1))
        if arrival_col != "None":
            data['Log_Arrivals'] = np.log(data[arrival_col].replace(0, np.nan)).fillna(method='bfill')
        data.dropna(inplace=True)

        # ===================== Analysis Tabs =====================
        tab1, tab2, tab3 = st.tabs(["📊 Risk Estimation", "🧪 Model Backtesting", "🔮 Future Forecast"])

        with tab1:
            st.subheader("Static Risk Metrics")
            mu, sigma = data['Log_Returns'].mean(), data['Log_Returns'].std()
            z_score = norm.ppf(1 - conf_level)

            # 1. Historical
            h_var = np.percentile(data['Log_Returns'], (1 - conf_level) * 100)
            
            # 2. Parametric
            p_var = mu + z_score * sigma
            
            # 3. Monte Carlo & CVaR (Supply-Sensitive if arrivals exist)
            if arrival_col != "None":
                X = sm.add_constant(data['Log_Arrivals'])
                res = sm.OLS(data['Log_Returns'], X).fit()
                sim_arr = np.random.normal(data['Log_Arrivals'].mean(), data['Log_Arrivals'].std(), num_sims)
                sim_mu = res.params[0] + res.params[1] * sim_arr
                sim_ret = np.random.normal(sim_mu, res.resid.std(), num_sims)
            else:
                sim_ret = np.random.normal(mu, sigma, num_sims)
            
            mc_var = np.percentile(sim_ret, (1 - conf_level) * 100)
            cvar = sim_ret[sim_ret <= mc_var].mean()

            # Display Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Historical VaR", f"{to_pct(h_var):.2f}%")
            m2.metric("Parametric VaR", f"{to_pct(p_var):.2f}%")
            m3.metric("Monte Carlo VaR", f"{to_pct(mc_var):.2f}%")
            m4.metric("Expected Shortfall (CVaR)", f"{to_pct(cvar):.2f}%")

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(sim_ret, kde=True, color='seagreen', ax=ax, label="Simulated Returns")
            ax.axvline(mc_var, color='red', linestyle='--', label=f'VaR ({to_pct(mc_var):.1f}%)')
            ax.axvline(cvar, color='purple', linestyle='--', label=f'CVaR ({to_pct(cvar):.1f}%)')
            ax.set_title("Return Distribution & Risk Thresholds")
            plt.legend()
            st.pyplot(fig)

            # Policy Suggestions
            st.markdown('<div class="report-box">', unsafe_allow_html=True)
            st.subheader("💡 Policy Recommendations")
            if to_pct(cvar) < -15:
                st.error("**High Risk Alert:** Prices are extremely sensitive to supply shocks. Recommendation: Strengthen Price Stabilization Fund (PSF) and promote farm-gate processing.")
            else:
                st.success("**Moderate Risk:** Market exhibits standard volatility. Recommendation: Improve real-time market information dissemination.")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.subheader("Kupiec POF Backtesting (Rolling)")
            if len(data) < window + 20:
                st.warning("Not enough data for rolling backtest.")
            else:
                bt_results = []
                progress_bar = st.progress(0)
                
                # Rolling Simulation
                for i in range(window, len(data)):
                    train = data.iloc[i-window:i]
                    actual = data['Log_Returns'].iloc[i]
                    
                    if arrival_col != "None":
                        model = sm.OLS(train['Log_Returns'], sm.add_constant(train['Log_Arrivals'])).fit()
                        s_arr = np.random.normal(train['Log_Arrivals'].mean(), train['Log_Arrivals'].std(), 1000)
                        s_mu = model.params[0] + model.params[1] * s_arr
                        s_ret = np.random.normal(s_mu, model.resid.std(), 1000)
                    else:
                        s_ret = np.random.normal(train['Log_Returns'].mean(), train['Log_Returns'].std(), 1000)
                    
                    v_limit = np.percentile(s_ret, (1 - conf_level) * 100)
                    bt_results.append({'Date': data.index[i], 'Actual': actual, 'VaR': v_limit, 'Breach': actual < v_limit})
                    if i % 10 == 0: progress_bar.progress((i - window) / (len(data) - window))
                
                bt_df = pd.DataFrame(bt_results).set_index('Date')
                total_obs = len(bt_df)
                total_breaches = bt_df['Breach'].sum()
                lr, p_val = kupiec_test(total_breaches, total_obs, 1 - conf_level)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Observed Breach Rate", f"{(total_breaches/total_obs)*100:.2f}%")
                c2.metric("Kupiec P-Value", f"{p_val:.4f}")
                c3.write("✅ Model Valid" if p_val > 0.05 else "❌ Model Rejected")

                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.plot(bt_df['Actual'], label="Actual Return", alpha=0.6)
                ax2.plot(bt_df['VaR'], color='red', linestyle='--', label="Estimated VaR")
                ax2.scatter(bt_df[bt_df['Breach']].index, bt_df[bt_df['Breach']]['Actual'], color='black', label="Breach")
                plt.legend()
                st.pyplot(fig2)

        with tab3:
            st.subheader("🔮 4-Week Supply-Sensitive Forecast")
            if arrival_col != "None":
                st.info("Simulating future prices based on historical arrival volatility.")
                # Latest parameters
                last_train = data.iloc[-window:]
                final_model = sm.OLS(last_train['Log_Returns'], sm.add_constant(last_train['Log_Arrivals'])).fit()
                
                f_weeks = [1, 2, 3, 4]
                f_risks = []
                for w in f_weeks:
                    f_arr = np.random.normal(last_train['Log_Arrivals'].mean(), last_train['Log_Arrivals'].std(), 5000)
                    f_mu = final_model.params[0] + final_model.params[1] * f_arr
                    f_ret = np.random.normal(f_mu, final_model.resid.std(), 5000)
                    f_risks.append(to_pct(np.percentile(f_ret, (1 - conf_level) * 100)))
                
                f_df = pd.DataFrame({"Week Ahead": f_weeks, "Potential Max Loss (%)": f_risks})
                st.table(f_df)
            else:
                st.warning("Please provide an Arrival column to enable supply-sensitive forecasting.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Waiting for data upload. Please ensure your CSV contains 'Date', 'Modal' (Price), and 'Arrivals'.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8rem;">
    RAISE_X v2.1 | PhD Thesis Resource Tool | Developer: Suman L | UAS Bengaluru
</div>
""", unsafe_allow_html=True)
