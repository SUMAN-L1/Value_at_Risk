# app.py — RAISE_X: Real-time Insights on Scenario Evaluation
# Fully automated, non-repetitive VaR/CVaR analysis for agricultural markets

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, chi2, jarque_bera
from jinja2 import Template
import warnings
import io

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY = True
except Exception:
    PLOTLY = False

# ─────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAISE_X | Market Risk Analysis",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0d2b1f 0%, #1a4731 50%, #0d2b1f 100%);
    padding: 2.5rem 2rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    border: 1px solid #2e6b46; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top:0; left:0; right:0; bottom:0;
    background: repeating-linear-gradient(45deg, transparent, transparent 40px,
        rgba(255,255,255,0.01) 40px, rgba(255,255,255,0.01) 80px);
}
.hero-title {
    font-family: 'Playfair Display', serif; font-size: 2.6rem; font-weight: 900;
    color: #e8f5e3; margin: 0; letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.1rem; color: #82c99a; margin: 0.4rem 0 0;
    font-weight: 300; letter-spacing: 2px; text-transform: uppercase;
}
.hero-badge {
    display: inline-block; background: #2e6b46; color: #a8e6be;
    font-size: 0.72rem; padding: 3px 10px; border-radius: 20px;
    margin-top: 0.8rem; letter-spacing: 1px; text-transform: uppercase; font-weight: 600;
}
.section-title {
    font-family: 'Playfair Display', serif; color: #1a4731; font-size: 1.5rem;
    font-weight: 700; padding: 0.5rem 0; border-bottom: 3px solid #2e6b46; margin: 1.5rem 0 1rem;
}
.risk-high { background:#fff0f0; border-left:5px solid #dc3545; padding:1rem 1.2rem; border-radius:10px; margin:0.5rem 0; }
.risk-mod  { background:#fffbf0; border-left:5px solid #fd7e14; padding:1rem 1.2rem; border-radius:10px; margin:0.5rem 0; }
.risk-low  { background:#f0faf4; border-left:5px solid #28a745; padding:1rem 1.2rem; border-radius:10px; margin:0.5rem 0; }
.info-box  { background:#f0f8ff; padding:1rem; border-radius:10px; border-left:5px solid #4682B4; margin:0.8rem 0; }
.warn-box  { background:#fff3cd; padding:1rem; border-radius:10px; border-left:5px solid #ffc107; margin:0.8rem 0; }
.ok-box    { background:#d4edda; padding:1rem; border-radius:10px; border-left:5px solid #28a745;  margin:0.8rem 0; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d2b1f 0%, #1a4731 100%); }
[data-testid="stSidebar"] * { color: #d4edd9 !important; }

.footer {
    text-align:center; padding:2rem;
    background: linear-gradient(135deg, #0d2b1f, #1a4731);
    border-radius:12px; margin-top:2rem; color:#82c99a;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🍅 RAISE_X</div>
    <div class="hero-subtitle">Real-time Insights on Scenario Evaluation</div>
    <div class="hero-badge">Market Risk Analysis for Farmers · v3.0</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")
    st.markdown("---")
    st.markdown("**📊 Confidence Level**")
    conf_pct = st.slider("Confidence (%)", 90, 99, 95, 1,
                         help="95% → 95% sure losses won't exceed predicted amount")
    confidence = conf_pct / 100.0

    st.markdown("**🎲 Monte Carlo Simulations**")
    n_sim = st.select_slider("Simulations", [1000, 2000, 5000, 10000, 20000, 50000], value=10000)

    st.markdown("**📅 Rolling Window (weeks)**")
    window = st.number_input("Weeks per estimation window", 10, 520, 52, 1,
                              help="52 = 1 year rolling window")

    st.markdown("**🔮 Forecast Horizon (weeks)**")
    horizon = st.number_input("Weeks ahead to forecast", 4, 104, 26, 1)

    st.markdown("---")
    st.markdown("**📈 Display**")
    use_plotly    = st.checkbox("Interactive charts", value=True and PLOTLY)
    show_forecast = st.checkbox("Future risk forecast", value=True)
    show_backtest = st.checkbox("Backtesting & Kupiec test", value=True)

    st.markdown("---")
    st.caption("💡 Start with defaults, then tune settings.")
    st.caption("Built by Suman L · UAS Bengaluru")

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def read_file(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    return pd.read_excel(f)

def prep_dates(df):
    for c in ["Date", "date", "DATE", "Dates", "dates"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=[c]).set_index(c).sort_index()
            return df
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df

def safe_log_arrivals(series):
    """Convert arrivals to log, replacing 0/NaN safely — compatible with all pandas versions."""
    return np.log(series.replace(0, np.nan)).bfill().ffill()

def to_pct(x):
    return (np.exp(x) - 1) * 100

def compute_var_metrics(series, conf, n_sims):
    """Compute Historical, Parametric, Monte Carlo VaR and CVaR from a price series."""
    log_ret = np.log(series / series.shift(1)).dropna()
    mu      = log_ret.mean()
    sigma   = log_ret.std()
    z       = norm.ppf(1 - conf)

    hist_var  = np.percentile(log_ret, (1 - conf) * 100)
    param_var = mu + z * sigma

    np.random.seed(42)
    sims    = np.random.normal(mu, sigma, int(n_sims))
    mc_var  = np.percentile(sims, (1 - conf) * 100)
    mc_cvar = sims[sims <= mc_var].mean()

    return dict(log_returns=log_ret, mu=mu, sigma=sigma,
                hist_var=hist_var, param_var=param_var,
                mc_var=mc_var, mc_cvar=mc_cvar, sims=sims)

def run_ols(train_df, returns_col="Log_Returns", arrivals_col="Log_Arrivals"):
    X = sm.add_constant(train_df[arrivals_col])
    y = train_df[returns_col]
    return sm.OLS(y, X).fit()

def kupiec_test(n_obs, n_breaches, conf):
    p   = 1 - conf
    p_h = n_breaches / n_obs if n_obs > 0 else 0.0
    def safe_log(v): return np.log(v) if v > 0 else -1e10
    ph  = min(max(p_h, 1e-10), 1 - 1e-10)
    ll0 = (n_obs - n_breaches) * safe_log(1 - p)  + n_breaches * safe_log(p)
    ll1 = (n_obs - n_breaches) * safe_log(1 - ph) + n_breaches * safe_log(ph)
    LR  = -2 * (ll0 - ll1)
    pv  = 1 - chi2.cdf(LR, df=1)
    return LR, pv

def descriptive_stats(log_ret):
    _, jb_p = jarque_bera(log_ret)
    return {
        "Observations":  len(log_ret),
        "Mean":          f"{log_ret.mean():.6f}",
        "Std Dev":       f"{log_ret.std():.6f}",
        "Skewness":      f"{log_ret.skew():.4f}",
        "Kurtosis":      f"{log_ret.kurtosis():.4f}",
        "Min":           f"{log_ret.min():.4f}",
        "Max":           f"{log_ret.max():.4f}",
        "Jarque-Bera p": f"{jb_p:.4f}",
    }

def risk_label(mc_cvar_log):
    v = abs(mc_cvar_log)
    if v > 0.10: return "HIGH",     "#dc3545"
    if v > 0.05: return "MODERATE", "#fd7e14"
    return "LOW", "#28a745"

def plot_var_distribution(log_ret, m, title, use_plotly):
    hist_var  = m["hist_var"]
    param_var = m["param_var"]
    mc_var    = m["mc_var"]
    mc_cvar   = m["mc_cvar"]

    if use_plotly and PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=log_ret, nbinsx=50,
                                   name="Empirical Returns", opacity=0.55,
                                   marker_color="#4a9e6e"))
        xs    = np.linspace(log_ret.min(), log_ret.max(), 300)
        scale = len(log_ret) * (log_ret.max() - log_ret.min()) / 50
        pdf   = norm.pdf(xs, m["mu"], m["sigma"]) * scale
        fig.add_trace(go.Scatter(x=xs, y=pdf, mode="lines",
                                  name="Parametric (Normal)",
                                  line=dict(color="#2196F3", width=2)))
        for val, col, name in [
            (hist_var,  "#222222", f"Historical VaR  {to_pct(hist_var):.2f}%"),
            (param_var, "#2196F3", f"Parametric VaR  {to_pct(param_var):.2f}%"),
            (mc_var,    "#e74c3c", f"MC VaR          {to_pct(mc_var):.2f}%"),
            (mc_cvar,   "#8e44ad", f"CVaR            {to_pct(mc_cvar):.2f}%"),
        ]:
            fig.add_vline(x=val, line=dict(color=col, dash="dash", width=2))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                      marker=dict(color=col, size=10), name=name))
        fig.update_layout(title=title, xaxis_title="Log Returns",
                           yaxis_title="Frequency", height=420,
                           plot_bgcolor="#f9fffe", paper_bgcolor="#f9fffe",
                           legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(log_ret, bins=50, kde=False, ax=ax, color="#4a9e6e", alpha=0.6)
        xs    = np.linspace(log_ret.min(), log_ret.max(), 300)
        scale = len(log_ret) * (log_ret.max() - log_ret.min()) / 50
        ax.plot(xs, norm.pdf(xs, m["mu"], m["sigma"]) * scale, "--",
                color="#2196F3", label="Parametric Normal")
        ax.axvline(hist_var,  color="#222222", ls="--", lw=2,
                   label=f"Historical VaR {to_pct(hist_var):.2f}%")
        ax.axvline(param_var, color="#2196F3", ls="--", lw=2,
                   label=f"Parametric VaR {to_pct(param_var):.2f}%")
        ax.axvline(mc_var,    color="#e74c3c", ls="--", lw=2,
                   label=f"MC VaR {to_pct(mc_var):.2f}%")
        ax.axvline(mc_cvar,   color="#8e44ad", ls="--", lw=2,
                   label=f"CVaR {to_pct(mc_cvar):.2f}%")
        ax.set_title(title); ax.set_xlabel("Log Returns")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

def policy_brief(market, conf, m, rl):
    h  = round(to_pct(m["hist_var"]),  2)
    p  = round(to_pct(m["param_var"]), 2)
    mc = round(to_pct(m["mc_var"]),    2)
    cv = round(to_pct(m["mc_cvar"]),   2)
    risk_css = "risk-high" if rl == "HIGH" else ("risk-mod" if rl == "MODERATE" else "risk-low")
    emoji    = "🔴" if rl == "HIGH" else ("🟡" if rl == "MODERATE" else "🟢")

    farmer_actions = {
        "HIGH": [
            "Stagger planting windows to avoid market gluts during peak supply",
            "Immediately seek FPO-led collective sales to institutional/export buyers",
            "Convert perishable surplus to storable value-added products",
            "Activate forward contracts — target locking in ≥40% of expected output",
            "Maintain emergency reserve fund equal to ≥15% of seasonal revenue",
        ],
        "MODERATE": [
            "Monitor mandi arrivals weekly; adjust sell timing to avoid peak-supply dips",
            "Explore FPO-assisted bulk supply agreements for price stability",
            "Keep 10–12 days of buffer stock capacity at farm or cold storage",
            "Plan crop diversification for next season to reduce single-crop exposure",
            "Build a contingency reserve of ~10% of seasonal revenue",
        ],
        "LOW": [
            "Continue standard agronomic and marketing practices",
            "Favourable window for input investment, variety trials, and expansion",
            "Maintain routine buffer stock (7 days); no emergency action required",
            "Document current price levels for use as baseline in future comparisons",
        ],
    }
    policy_actions = [
        "**Price Stabilization Fund** — Deploy targeted procurement/buffer stock releases during glut periods",
        "**Value-Addition Incentives** — Subsidise decentralised processing units (paste, ketchup, dehydration) near production clusters",
        "**Market Intelligence System** — Real-time mandi price & arrival dashboards disseminated to farmers via SMS/app",
        "**Crop-Insurance Expansion** — Extend weather + price-linked insurance products to cover tail-risk losses",
        "**Infrastructure Investment** — Cold-chain & grading centres to reduce distress sales during harvest surpluses",
    ]

    brief = f"""
<div class="{risk_css}">
<strong>{emoji} Risk Level: {rl}</strong> &nbsp;|&nbsp; Market: <strong>{market}</strong> &nbsp;|&nbsp; Confidence: <strong>{int(conf*100)}%</strong>
</div>

**📊 Risk Metrics Summary**

| Method | Log Return | Weekly Loss (%) |
|---|---|---|
| Historical VaR  | {m['hist_var']:.6f} | {h:.2f}% |
| Parametric VaR  | {m['param_var']:.6f} | {p:.2f}% |
| Monte Carlo VaR | {m['mc_var']:.6f} | {mc:.2f}% |
| Conditional CVaR| {m['mc_cvar']:.6f} | {cv:.2f}% |

**🔍 Statistical Profile**

| Statistic | Value |
|---|---|
| Weekly Return Mean | {m['mu']:.6f} |
| Std Deviation      | {m['sigma']:.6f} |
| Z-score ({int(conf*100)}% CI) | {norm.ppf(1 - conf):.3f} |

**🌾 Recommended Actions — Farmers**
"""
    for i, a in enumerate(farmer_actions[rl], 1):
        brief += f"\n{i}. {a}"
    brief += "\n\n**🏛️ Policy Recommendations**\n"
    for i, a in enumerate(policy_actions, 1):
        brief += f"\n{i}. {a}"
    return brief

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📁 Upload Market Data</div>', unsafe_allow_html=True)

col_up1, col_up2 = st.columns([2, 1])
with col_up1:
    uploaded = st.file_uploader(
        "Upload CSV, XLS, or XLSX (must have a Date column + at least one numeric price column)",
        type=["csv", "xls", "xlsx"]
    )
with col_up2:
    st.markdown("""
    <div class="info-box">
    <strong>Required columns</strong><br>
    • <code>Date</code> — weekly date column<br>
    • Any numeric price column (e.g. <code>Modal</code>)<br><br>
    <strong>For advanced analysis also include:</strong><br>
    • <code>Modal</code> — modal/market price<br>
    • <code>Arrivals</code> — weekly market arrivals
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────
with st.expander("📊 No data? Generate & download sample data"):
    s_market = st.text_input("Market name for sample", "Kolar")
    if st.button("Generate Sample Data", type="primary"):
        np.random.seed(42)
        dates  = pd.date_range("2015-01-04", "2024-12-29", freq="W-SUN")
        t      = np.arange(len(dates))
        modal  = np.maximum(50 + 12*np.sin(2*np.pi*t/52) + 0.08*t + np.random.normal(0, 6, len(dates)), 15)
        arr    = np.maximum(1000 - 220*np.sin(2*np.pi*t/52 + np.pi/4) + np.random.normal(0, 120, len(dates)), 80)
        sdf    = pd.DataFrame({"Date": dates, "Modal": modal, "Arrivals": arr,
                                "Wholesale": modal*0.78, "Retail": modal*1.32})
        st.dataframe(sdf.head(8))
        st.download_button("⬇️ Download sample CSV",
                            sdf.to_csv(index=False).encode(),
                            f"sample_{s_market.lower()}_weekly.csv", "text/csv")

if not uploaded:
    st.markdown("""
    <div class="info-box">
    <strong>👆 Upload your file above to begin analysis.</strong><br>
    Results will include: Historical VaR · Parametric VaR · Monte Carlo VaR · CVaR ·
    Backtesting · Kupiec Test · Future Risk Forecast · Policy Recommendations.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# LOAD & VALIDATE FILE
# ─────────────────────────────────────────────
try:
    df_raw = read_file(uploaded)
    df_raw = prep_dates(df_raw)
except Exception as e:
    st.error(f"❌ Could not read file: {e}")
    st.stop()

numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.error("❌ No numeric columns found. Please check your file.")
    st.stop()

# ─────────────────────────────────────────────
# CONFIGURE ANALYSIS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🎛️ Configure Analysis</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    d0, d1 = df_raw.index.min().date(), df_raw.index.max().date()
    dr = st.date_input("Date range", [d0, d1])
with c2:
    default_idx = numeric_cols.index("Modal") if "Modal" in numeric_cols else 0
    price_col   = st.selectbox("Price column to analyse", numeric_cols, index=default_idx)
with c3:
    extra_cols   = [c for c in numeric_cols if c != price_col]
    compare_cols = st.multiselect("Also compare columns (optional)", extra_cols)

if len(dr) == 2:
    df = df_raw.loc[str(dr[0]):str(dr[1])].copy()
else:
    df = df_raw.copy()

if len(df) < 40:
    st.warning("⚠️ Fewer than 40 rows after filtering — results may be unreliable.")

# Data overview
st.markdown('<div class="ok-box">✅ <strong>Data loaded successfully.</strong></div>',
            unsafe_allow_html=True)
ov1, ov2, ov3, ov4 = st.columns(4)
ov1.metric("Rows",    len(df))
ov2.metric("Columns", len(df.columns))
ov3.metric("Start",   str(df.index.min().date()))
ov4.metric("End",     str(df.index.max().date()))

with st.expander("📋 Preview raw data"):
    st.dataframe(df.head(20), use_container_width=True)

# ─────────────────────────────────────────────
# PER-COLUMN VaR ANALYSIS
# ─────────────────────────────────────────────
all_cols   = [price_col] + compare_cols
summary_rows = []

for col in all_cols:
    series = df[col].copy().replace(0, np.nan).dropna()
    if len(series) < 30:
        st.warning(f"⚠️ {col}: too few data points (need ≥ 30). Skipped.")
        continue

    st.markdown(f'<div class="section-title">📈 Analysis — {col}</div>', unsafe_allow_html=True)

    m  = compute_var_metrics(series, confidence, n_sim)
    lr = m["log_returns"]
    rl, rl_color = risk_label(m["mc_cvar"])

    # Descriptive stats
    with st.expander(f"📐 Descriptive Statistics — {col}"):
        st.table(pd.DataFrame.from_dict(descriptive_stats(lr),
                                         orient="index", columns=["Value"]))

    # Price series chart
    if use_plotly and PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                                  name="Price", line=dict(color="#2e6b46", width=2)))
        fig.update_layout(title=f"Weekly Price Series — {col}",
                           xaxis_title="Date", yaxis_title="Price (₹/Quintal)",
                           height=300, plot_bgcolor="#f9fffe", paper_bgcolor="#f9fffe")
        st.plotly_chart(fig, use_container_width=True)

    # VaR distribution plot
    plot_var_distribution(lr, m, f"Return Distribution & Risk Metrics — {col}", use_plotly)

    # Risk metrics table
    st.markdown("#### 📊 Risk Metrics Table")
    rm_df = pd.DataFrame([{
        "Method":           meth,
        "Log Return":       f"{val:.6f}",
        "Weekly Loss (%)":  f"{to_pct(val):.3f}%",
        "Annualised (%)":   f"{to_pct(val) * np.sqrt(52):.2f}%",
    } for meth, val in [
        ("Historical VaR",   m["hist_var"]),
        ("Parametric VaR",   m["param_var"]),
        ("Monte Carlo VaR",  m["mc_var"]),
        ("Conditional CVaR", m["mc_cvar"]),
    ]])
    st.dataframe(rm_df, use_container_width=True, hide_index=True)

    summary_rows.append({
        "Market / Column": col,
        "Risk Level":      rl,
        "Hist VaR (%)":    round(to_pct(m["hist_var"]),  3),
        "Param VaR (%)":   round(to_pct(m["param_var"]), 3),
        "MC VaR (%)":      round(to_pct(m["mc_var"]),    3),
        "CVaR (%)":        round(to_pct(m["mc_cvar"]),   3),
        "Mean Return":     round(m["mu"],    6),
        "Std Dev":         round(m["sigma"], 6),
    })

    with st.expander(f"📝 Policy Brief & Recommendations — {col}"):
        st.markdown(policy_brief(col, confidence, m, rl))

    st.markdown("---")

# ─────────────────────────────────────────────
# CROSS-COLUMN COMPARISON
# ─────────────────────────────────────────────
if len(summary_rows) > 1:
    st.markdown('<div class="section-title">🔀 Cross-Column Risk Comparison</div>',
                unsafe_allow_html=True)
    sum_df = pd.DataFrame(summary_rows)
    st.dataframe(sum_df, use_container_width=True, hide_index=True)

    if use_plotly and PLOTLY:
        fig = go.Figure()
        for method, clr in [("Hist VaR (%)","#4a9e6e"),
                              ("MC VaR (%)", "#e74c3c"),
                              ("CVaR (%)",   "#8e44ad")]:
            fig.add_trace(go.Bar(name=method,
                                  x=sum_df["Market / Column"],
                                  y=sum_df[method].abs(),
                                  marker_color=clr))
        fig.update_layout(barmode="group", title="Risk Comparison Across Columns",
                           yaxis_title="Absolute Weekly Loss (%)", height=380,
                           plot_bgcolor="#f9fffe", paper_bgcolor="#f9fffe")
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# ADVANCED ANALYSIS  (requires Modal + Arrivals)
# ─────────────────────────────────────────────
has_adv = "Modal" in df.columns and "Arrivals" in df.columns

if not has_adv:
    st.markdown("""<div class="info-box">
    <strong>ℹ️ Advanced Analysis Unavailable</strong><br>
    To enable OLS regression, backtesting, Kupiec test and forecasting,
    your file must contain <code>Modal</code> and <code>Arrivals</code> columns.
    </div>""", unsafe_allow_html=True)
else:
    # ── Prepare advanced dataset (pandas-version-safe) ──
    ds = df[["Modal", "Arrivals"]].copy()
    ds["Log_Returns"]  = np.log(ds["Modal"] / ds["Modal"].shift(1))
    ds["Log_Arrivals"] = safe_log_arrivals(ds["Arrivals"])   # ← uses bfill().ffill()
    ds = ds.dropna()

    # ── OLS Regression ──
    st.markdown('<div class="section-title">📉 OLS Regression: Returns ~ Log Arrivals</div>',
                unsafe_allow_html=True)
    ols  = run_ols(ds)
    corr = ds[["Log_Returns", "Log_Arrivals"]].corr().iloc[0, 1]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Intercept (α)",   f"{ols.params.iloc[0]:.6f}")
    r2.metric("Coefficient (β)", f"{ols.params.iloc[1]:.6f}")
    r3.metric("R²",              f"{ols.rsquared:.4f}")
    r4.metric("Correlation",     f"{corr:.4f}")

    with st.expander("📋 Full OLS Regression Summary"):
        st.text(ols.summary().as_text())

    # ── BACKTESTING ──
    if show_backtest and len(ds) > window:
        st.markdown('<div class="section-title">🔬 Rolling Backtest — Monte Carlo VaR & CVaR</div>',
                    unsafe_allow_html=True)

        np.random.seed(42)
        bt_rows   = []
        prog      = st.progress(0, "Running rolling backtest…")
        n_steps   = len(ds) - int(window)

        for i in range(int(window), len(ds)):
            train  = ds.iloc[i - int(window): i]
            test   = ds.iloc[i]
            mdl    = run_ols(train)
            res_s  = mdl.resid.std()
            sim_a  = np.random.normal(train["Log_Arrivals"].mean(),
                                       train["Log_Arrivals"].std(), int(n_sim))
            sim_m  = mdl.params.iloc[0] + mdl.params.iloc[1] * sim_a
            sim_r  = np.random.normal(sim_m, res_s, int(n_sim))
            vl     = np.percentile(sim_r, (1 - confidence) * 100)
            cv     = sim_r[sim_r <= vl].mean()
            actual = test["Log_Returns"]
            bt_rows.append({
                "Date":          ds.index[i],
                "Actual_Return": actual,
                "MC_VaR":        vl,
                "MC_CVaR":       cv,
                "Breach_VaR":    bool(actual < vl),
                "Breach_CVaR":   bool(actual < cv),
            })
            prog.progress((i - int(window) + 1) / n_steps)

        prog.empty()
        bt = pd.DataFrame(bt_rows).set_index("Date")

        # Kupiec test
        n, x         = len(bt), int(bt["Breach_VaR"].sum())
        LR, kp       = kupiec_test(n, x, confidence)
        breach_rate_var  = x / n * 100
        breach_rate_cvar = bt["Breach_CVaR"].mean() * 100
        expected_rate    = (1 - confidence) * 100

        # Backtest plot
        act_pct  = (np.exp(bt["Actual_Return"]) - 1) * 100
        var_pct  = (np.exp(bt["MC_VaR"]) - 1)  * 100
        cvar_pct = (np.exp(bt["MC_CVaR"]) - 1) * 100

        if use_plotly and PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt.index, y=act_pct,  mode="lines",
                                      name="Actual Returns (%)",
                                      line=dict(color="#2e6b46", width=1.5)))
            fig.add_trace(go.Scatter(x=bt.index, y=var_pct,  mode="lines",
                                      name="MC VaR (%)",
                                      line=dict(color="#e74c3c", dash="dash")))
            fig.add_trace(go.Scatter(x=bt.index, y=cvar_pct, mode="lines",
                                      name="CVaR (%)",
                                      line=dict(color="#8e44ad", dash="dot")))
            breach_dates = bt[bt["Breach_VaR"]].index
            if len(breach_dates):
                fig.add_trace(go.Scatter(
                    x=breach_dates,
                    y=(np.exp(bt.loc[breach_dates, "Actual_Return"]) - 1) * 100,
                    mode="markers", name="VaR Breach",
                    marker=dict(color="#ff6600", size=7, symbol="x")))
            fig.update_layout(
                title=f"Rolling Backtest — Kupiec LR={LR:.3f}, p={kp:.4f}",
                xaxis_title="Date", yaxis_title="Weekly Return (%)",
                height=420, plot_bgcolor="#f9fffe", paper_bgcolor="#f9fffe",
                hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(bt.index, act_pct,  label="Actual",  color="#2e6b46", lw=1.5)
            ax.plot(bt.index, var_pct,  label="MC VaR",  color="#e74c3c", ls="--")
            ax.plot(bt.index, cvar_pct, label="CVaR",    color="#8e44ad", ls="-.")
            breaches = bt[bt["Breach_VaR"]]
            if not breaches.empty:
                ax.scatter(breaches.index,
                            (np.exp(breaches["Actual_Return"]) - 1) * 100,
                            color="#ff6600", marker="x", zorder=5, label="VaR Breach")
            ax.set_title(f"Rolling Backtest — Kupiec LR={LR:.3f} p={kp:.4f}")
            ax.set_ylabel("Weekly Return (%)"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close()

        # Backtest metrics
        st.markdown("#### 📊 Backtest Performance Metrics")
        bm1, bm2, bm3, bm4, bm5 = st.columns(5)
        bm1.metric("Total Periods",              n)
        bm2.metric("VaR Breaches",               int(x))
        bm3.metric("Observed Breach Rate (VaR)", f"{breach_rate_var:.2f}%")
        bm4.metric("Expected Breach Rate",       f"{expected_rate:.2f}%")
        bm5.metric("Kupiec p-value",             f"{kp:.4f}")

        if kp < 0.05:
            st.markdown("""<div class="warn-box">
            ⚠️ <strong>Kupiec Test: REJECT H₀</strong> — Observed breach rate significantly differs
            from expected. Model may be mis-calibrated.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="ok-box">
            ✅ <strong>Kupiec Test: ACCEPT H₀</strong> — Observed breach rate is statistically
            consistent with expected failure rate. Model is well-calibrated.
            </div>""", unsafe_allow_html=True)

        if abs(breach_rate_cvar - expected_rate) < 2:
            st.markdown('<div class="ok-box">✅ <strong>CVaR Calibration:</strong> '
                        'Breach rate closely matches nominal rate.</div>',
                        unsafe_allow_html=True)
        elif breach_rate_cvar > expected_rate + 2:
            st.markdown('<div class="warn-box">🔴 <strong>CVaR underestimating tail risk</strong>'
                        ' — actual losses exceed model predictions.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-box">🟢 <strong>CVaR appears conservative</strong>'
                        ' — model is over-predicting losses (safe side).</div>',
                        unsafe_allow_html=True)

        bmc1, bmc2, bmc3 = st.columns(3)
        bmc1.metric("CVaR Breach Rate",   f"{breach_rate_cvar:.1f}%")
        bmc2.metric("Expected CVaR Rate", f"{expected_rate:.1f}%")
        bmc3.metric("Model Calibration",  f"{100 - abs(breach_rate_cvar - expected_rate):.1f}%")

    # ── FUTURE RISK FORECAST ──
    if show_forecast:
        st.markdown('<div class="section-title">🔮 Future Risk Forecast</div>',
                    unsafe_allow_html=True)

        train_latest = ds.iloc[-int(window):]
        mdl_latest   = run_ols(train_latest)
        mu_arr       = train_latest["Log_Arrivals"].mean()
        sig_arr      = train_latest["Log_Arrivals"].std()
        res_sig      = mdl_latest.resid.std()

        fc_rows = []
        for h in range(1, int(horizon) + 1):
            np.random.seed(42 + h)
            sim_a = np.random.normal(mu_arr, sig_arr, int(n_sim))
            sim_m = mdl_latest.params.iloc[0] + mdl_latest.params.iloc[1] * sim_a
            sim_r = np.random.normal(sim_m, res_sig, int(n_sim))
            vl    = np.percentile(sim_r, (1 - confidence) * 100)
            cv    = sim_r[sim_r <= vl].mean()
            fc_rows.append({"Week": h,
                              "VaR (log)":  vl,  "CVaR (log)": cv,
                              "VaR (%)":  to_pct(vl), "CVaR (%)": to_pct(cv)})

        fc_df   = pd.DataFrame(fc_rows).set_index("Week")
        worst_w = int(fc_df["CVaR (%)"].idxmin())
        worst_v = fc_df["CVaR (%)"].min()
        avg_cv  = fc_df["CVaR (%)"].mean()
        first5  = fc_df["CVaR (%)"].iloc[:min(5, len(fc_df))].mean()
        last5   = fc_df["CVaR (%)"].iloc[max(0, len(fc_df) - 5):].mean()
        trend   = "📈 INCREASING" if last5 < first5 else "📉 STABLE / DECLINING"

        f1, f2, f3 = st.columns(3)
        f1.metric("Worst Risk Week",       f"Week {worst_w}")
        f2.metric("Max CVaR Loss",         f"{worst_v:.2f}%")
        f3.metric("Avg CVaR over Horizon", f"{avg_cv:.2f}%")
        st.caption(f"Risk Trend: {trend}")

        if use_plotly and PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df["CVaR (%)"],
                                      mode="lines+markers", name="CVaR Forecast",
                                      line=dict(color="#e74c3c", width=3)))
            fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df["VaR (%)"],
                                      mode="lines+markers", name="VaR Forecast",
                                      line=dict(color="#f39c12", width=2)))
            fig.add_vline(x=worst_w, line=dict(color="#e74c3c", dash="dash"),
                           annotation_text=f"Worst Week {worst_w}")
            fig.add_hrect(y0=fc_df["CVaR (%)"].min(), y1=0,
                           fillcolor="rgba(231,76,60,0.07)", line_width=0)
            fig.update_layout(title=f"Risk Forecast — Next {horizon} Weeks",
                               xaxis_title="Weeks Ahead",
                               yaxis_title="Potential Weekly Loss (%)",
                               height=400, plot_bgcolor="#f9fffe", paper_bgcolor="#f9fffe")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(fc_df.index, fc_df["CVaR (%)"], marker="o", lw=2,
                    label="CVaR", color="#e74c3c")
            ax.plot(fc_df.index, fc_df["VaR (%)"],  marker="s", lw=2,
                    label="VaR",  color="#f39c12")
            ax.axvline(worst_w, color="#e74c3c", ls="--",
                        label=f"Worst Week {worst_w}")
            ax.fill_between(fc_df.index, fc_df["CVaR (%)"], -50,
                             alpha=0.08, color="#e74c3c")
            ax.set_xlabel("Weeks Ahead"); ax.set_ylabel("Potential Weekly Loss (%)")
            ax.set_title(f"Risk Forecast — Next {horizon} Weeks")
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close()

        with st.expander("📋 Full Forecast Table"):
            disp = fc_df[["VaR (%)", "CVaR (%)"]].copy().round(3)
            disp["VaR (%)"]  = disp["VaR (%)"].astype(str)  + "%"
            disp["CVaR (%)"] = disp["CVaR (%)"].astype(str) + "%"
            st.dataframe(disp, use_container_width=True)

        if worst_v < -10:
            st.markdown(f"""<div class="risk-high">
            🚨 <strong>HIGH ALERT — Week {worst_w}</strong><br>
            Expected worst-case weekly loss exceeds <strong>10%</strong>.
            Immediate risk mitigation required.<br>
            • Reduce perishable inventory · Activate forward contracts · Notify FPO immediately
            </div>""", unsafe_allow_html=True)
        elif worst_v < -5:
            st.markdown(f"""<div class="risk-mod">
            ⚠️ <strong>CAUTION — Week {worst_w}</strong><br>
            Moderate tail risk (5–10% loss). Monitor arrivals, maintain buffer stock,
            keep backup buyers ready.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="risk-low">
            ✅ <strong>NORMAL CONDITIONS</strong><br>
            Risk levels are low over the forecast horizon.
            Good time for expansion and investment planning.
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FINAL SUMMARY TABLE + DOWNLOAD
# ─────────────────────────────────────────────
if summary_rows:
    st.markdown('<div class="section-title">📋 Overall Risk Summary</div>',
                unsafe_allow_html=True)
    final_df = pd.DataFrame(summary_rows)
    st.dataframe(final_df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download Risk Summary CSV",
                        final_df.to_csv(index=False).encode(),
                        "risk_summary.csv", "text/csv")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;color:#a8e6be;margin-bottom:0.5rem;">
        🌾 RAISE_X — Built for Farmers, By a Young Economist
    </div>
    <div style="font-size:0.9rem;color:#82c99a;margin-bottom:1rem;">
        Applies financial risk econometrics (VaR, CVaR, Monte Carlo, OLS, Kupiec backtesting)
        to agricultural commodity markets to support data-driven farming decisions.
    </div>
    <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;font-size:0.9rem;">
        <span>📧 sumanecon.uas@outlook.in</span>
        <span>👨‍💼 Suman L</span>
        <span>🎓 UAS Bengaluru</span>
    </div>
    <div style="font-size:0.75rem;color:#5a9a6a;margin-top:1rem;">
        Version 3.0 · RAISE_X · Scientific Market Risk Analysis
    </div>
</div>
""", unsafe_allow_html=True)
