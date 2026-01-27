import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION & DATA ---
# Optimized Weights from Task 4
# Note: These values are derived from the 'scipy.optimize' run in Task 4
PORTFOLIOS = {
    "Growth (Max Sharpe Ratio)": {
        "Description": "Maximizes return per unit of risk. Heavily favors equities (SPY). Best for long-term investors.",
        "Expected Return": "15.55%",
        "Expected Volatility": "12.30%",
        "Sharpe Ratio": "1.26",
        "Weights": {"TSLA": 0.0000, "BND": 0.0000, "SPY": 1.0000}
    },
    "Stability (Min Volatility)": {
        "Description": "Minimizes daily price fluctuations. Heavily favors bonds (BND). Best for risk-averse or short-term investors.",
        "Expected Return": "2.67%",
        "Expected Volatility": "4.01%",
        "Sharpe Ratio": "0.67",
        "Weights": {"TSLA": 0.0145, "BND": 0.9422, "SPY": 0.0433}
    }
}

# --- APP LAYOUT ---
st.set_page_config(page_title="Robo-Advisor Interface", page_icon="📈")

st.title("📈 AI Investment Advisor")
st.markdown("""
Welcome to the Time Series Forecasting & Portfolio Optimization Robo-Advisor.
Enter your investment amount below to see our AI-driven recommendation based on the **Jan 2025 - Jan 2026** market forecast.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Investment Inputs")
amount = st.sidebar.number_input("Total Investment ($)", min_value=100.0, value=10000.0, step=100.0)
strategy_name = st.sidebar.selectbox("Choose Your Strategy", list(PORTFOLIOS.keys()))

# --- MAIN LOGIC ---
selected_portfolio = PORTFOLIOS[strategy_name]
weights = selected_portfolio["Weights"]

# Calculate Allocation
allocation = {asset: amt * amount for asset, amt in weights.items()}
allocation_df = pd.DataFrame(list(allocation.items()), columns=["Asset", "Allocation ($)"])
allocation_df["Weight (%)"] = [f"{w:.2%}" for w in weights.values()]

# --- DISPLAY RESULTS ---
st.subheader(f"Strategy: {strategy_name}")
st.info(selected_portfolio["Description"])

# Metrics Column
col1, col2, col3 = st.columns(3)
col1.metric("Expected Annual Return", selected_portfolio["Expected Return"])
col2.metric("Expected Volatility", selected_portfolio["Expected Volatility"])
col3.metric("Sharpe Ratio", selected_portfolio["Sharpe Ratio"])

st.divider()

# Allocation Table & Chart
col_table, col_chart = st.columns([1, 1])

with col_table:
    st.markdown("### Asset Allocation Breakdown")
    st.dataframe(allocation_df.set_index("Asset").style.format({"Allocation ($)": "${:,.2f}"}))

with col_chart:
    st.markdown("### Visual Breakdown")
    # Donut Chart
    fig, ax = plt.subplots()
    filtered_weights = {k: v for k, v in weights.items() if v > 0.005} # Filter purely for display (show > 0.5%)
    
    if filtered_weights:
        ax.pie(filtered_weights.values(), labels=filtered_weights.keys(), autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
        ax.axis('equal') 
        st.pyplot(fig)
    else:
        st.write("Allocation is 100% in a single asset.")

# --- CONCLUSION ---
st.divider()
st.markdown("### 🤖 Review")
if strategy_name == "Growth (Max Sharpe Ratio)":
    st.success(f"For an investment of **${amount:,.2f}**, we recommend a **100% Equity** allocation (SPY). Our forecast suggests TSLA's volatility outweighs its potential return this year.")
else:
    st.warning(f"For an investment of **${amount:,.2f}**, we recommend a **Conservative** allocation focusing on BND (Bonds), with minor exposure to SPY and TSLA to hedge inflation.")
