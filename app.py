import streamlit as st
import pandas as pd
#from simulator.variables import get_default_inputs
from simulator.demand import compute_demand
from simulator.calculus import compute_optimal_price
from simulator.logic import evaluate_price_optimality, generate_recommendation
from simulator.visualization import plot_revenue_profit
from utils.format import format_currency

st.set_page_config(page_title="Dynamic Pricing Simulator", layout="wide")
st.title("📊 Dynamic Pricing Simulator")
st.markdown("Use calculus-powered logic to optimize your product's pricing strategy.")
with st.sidebar:
    st.header("Input Parameters")
    
    demand_type = st.selectbox("Demand Function", ["Linear", "Exponential"])
    a = st.number_input("Demand Coefficient a", min_value=0, value=1000)
    b = st.number_input("Demand Coefficient b", min_value=0.1, value=15.0)
    price = st.slider("Initial Price ($)", 1, 100, 50)
    fixed_cost = st.number_input("Fixed Cost ($)", min_value=0, value=2000)
    variable_cost = st.number_input("Variable Cost per Unit ($)", min_value=0, value=5)
    run_simulation = st.button("Run Simulation")

if run_simulation:
    demand_fn, prices, demand, revenue, profit, cost = compute_demand(demand_type, a, b, fixed_cost, variable_cost)
    optimal_price, max_profit, demand_optimal = compute_optimal_price(demand_type,demand_fn, fixed_cost, variable_cost,a,b)
    is_optimal = evaluate_price_optimality(price, optimal_price)
    recommendation = generate_recommendation(is_optimal, price, optimal_price)

    # Display numeric output
    st.subheader("Results")
    st.markdown(f"- **Input Price:** {format_currency(price)}")
    st.markdown(f"- **Optimal Price:** {format_currency(optimal_price)}")
    st.markdown(f"- **Max Profit:** {format_currency(max_profit)}")

    # Display recommendation
    st.warning(recommendation if not is_optimal else "✅ Your price is optimal or very close to it!")

    # Graphs
    st.pyplot(plot_revenue_profit(prices, revenue, profit, optimal_price, price))

uploaded_file = st.file_uploader("Upload Sample Pricing Data (CSV)", type=["csv"])
if uploaded_file:
    st.subheader("📂 Uploaded Data")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    # Extract columns
    try:
        prices = df["Price ($)"]
        demand = df["Demand (Units)"]
        revenue = df["Revenue ($)"]
        cost = df["Total Cost ($)"]
        profit = df["Profit ($)"]

        # Find optimal price in uploaded data
        max_profit_idx = profit.idxmax()
        optimal_price_from_csv = prices[max_profit_idx]
        max_profit_value = profit[max_profit_idx]

        # Allow user to enter a test price from range
        user_price_csv = st.number_input("Your Price (from uploaded range)", min_value=float(prices.min()), max_value=float(prices.max()), value=float(prices.iloc[0]))

        is_optimal_csv = evaluate_price_optimality(user_price_csv, optimal_price_from_csv)
        recommendation_csv = generate_recommendation(is_optimal_csv, user_price_csv, optimal_price_from_csv)

        # Display results
        st.subheader("CSV Data Analysis")
        st.markdown(f"- **Optimal Price (from CSV):** {format_currency(optimal_price_from_csv)}")
        st.markdown(f"- **Max Profit:** {format_currency(max_profit_value)}")
        st.markdown(f"- **Your Price:** {format_currency(user_price_csv)}")
        st.warning(recommendation_csv if not is_optimal_csv else "✅ Your price is optimal or very close to it!")

        # Plot uploaded data
        st.pyplot(plot_revenue_profit(prices, revenue, profit, optimal_price_from_csv, user_price_csv))

    except KeyError as e:
        st.error(f"Missing required column in CSV: {e}")
