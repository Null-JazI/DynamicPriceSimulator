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
    optimal_price, max_profit = compute_optimal_price(demand_type,demand_fn, fixed_cost, variable_cost,a,b)
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
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    # Future: Add analysis from uploaded file
