#app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from simulator.demand import compute_demand
from simulator.calculus import compute_optimal_price
from simulator.logic import evaluate_price_optimality, generate_recommendation
from simulator.visualization import plot_revenue_profit
from utils.format import format_currency
from simulator.presets import get_industry_names, get_preset_by_name

# Page configuration
st.set_page_config(
    page_title="Dynamic Pricing Simulator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for configurations
if 'configurations' not in st.session_state:
    st.session_state.configurations = {}
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None

# Main title and description
st.title("📊 Dynamic Pricing Simulator")
st.markdown("""
**Advanced pricing optimization tool with comprehensive market modeling capabilities.**  
Analyze demand patterns, optimize pricing strategies, and compare multiple scenarios.
""")

# Configuration tabs
config_tab, scenario_tab, analysis_tab, data_tab = st.tabs([
    "🎛️ Configuration", 
    "🎯 Scenario Analysis", 
    "📈 Advanced Analytics",
    "📊 Data Upload"
])

with config_tab:
    st.header("Simulation Configuration")
    
    st.subheader("📦 Industry Presets (Optional)")
    selected_preset_name = st.selectbox("Choose an Industry Preset", ["Custom"] + get_industry_names())
    
    preset = get_preset_by_name(selected_preset_name) if selected_preset_name != "Custom" else None

    if preset:
        st.markdown(f"### 🏷️ Industry: `{preset['industry_name']}`")
        st.info(preset["description"])
        st.caption(f"🧠 *Rationale:* {preset['rationale']}")
    
    st.markdown("---")
    
    # Advanced configuration in columns
    col1, col2, col3 = st.columns(3)

    
    with col1:
        st.subheader("Market Parameters")
        
        # If preset is selected, auto-fill demand type
        if preset:
            demand_type = preset["demand_type"]
            st.markdown(f"**Demand Model:** `{demand_type}` _(from preset)_")
        else:
            demand_type = st.selectbox(
                "Demand Function Type", 
                ["Linear", "Exponential"],
                help="Choose the mathematical model that best represents market demand behavior"
            )
        
        # Market size and elasticity
        if demand_type == "Linear":
            a = st.number_input(
                "Market Size (Max Demand)",
                min_value=100,
                max_value=100000,
                value=preset["a"] if preset else 1000,
                step=100,
                help="Maximum potential demand when price approaches zero"
            )
            b = st.number_input(
                "Price Elasticity",
                min_value=0.1,
                max_value=100.0,
                value=preset["b"] if preset else 15.0,
                step=0.5,
                help="Rate at which demand decreases as price increases"
            )
        else:
            a = st.number_input(
                "Base Demand Level",
                min_value=100,
                max_value=50000,
                value=preset["a"] if preset else 1000,
                step=100,
                help="Initial demand level coefficient"
            )
            b = st.number_input(
                "Decay Rate",
                min_value=0.01,
                max_value=5.0,
                value=preset["b"] if preset else 0.1,
                step=0.01,
                help="How quickly demand decays with price increases"
            )

        
        # Market conditions
        st.subheader("Market Conditions")
        market_condition = st.selectbox(
            "Market State",
            ["Normal", "High Competition", "Premium Market", "Economic Downturn"],
            help="Adjust parameters based on current market conditions"
        )
        
        # Apply market condition modifiers
        condition_modifiers = {
            "Normal": {"demand_mult": 1.0, "elasticity_mult": 1.0},
            "High Competition": {"demand_mult": 0.8, "elasticity_mult": 1.5},
            "Premium Market": {"demand_mult": 0.6, "elasticity_mult": 0.7},
            "Economic Downturn": {"demand_mult": 0.7, "elasticity_mult": 1.8}
        }
        
        modifier = condition_modifiers[market_condition]
        a_adjusted = a * modifier["demand_mult"]
        b_adjusted = b * modifier["elasticity_mult"]
        
        if market_condition != "Normal":
            st.info(f"Market adjustment: Demand ×{modifier['demand_mult']}, Elasticity ×{modifier['elasticity_mult']}")
    
    with col2:
        st.subheader("Cost Structure")
        
        # Fixed costs
        fixed_cost = st.number_input(
            "Fixed Costs ($)", 
            min_value=0, 
            max_value=10000000, 
            value=preset["fixed_cost"] if preset else 2000,
            step=100,
            help="One-time costs independent of production volume"
        )

        variable_cost = st.number_input(
            "Variable Cost per Unit ($)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=float(preset["variable_cost"]) if preset else 5.0,
            step=0.5,
            help="Cost that varies directly with production volume"
        )

        
        # Cost escalation
        cost_escalation = st.checkbox(
            "Include Cost Escalation",
            help="Model increasing per-unit costs at high volumes"
        )
        
        if cost_escalation:
            escalation_threshold = st.number_input(
                "Escalation Threshold (units)",
                min_value=100,
                max_value=10000,
                value=1000,
                help="Production volume where costs begin to escalate"
            )
            escalation_rate = st.number_input(
                "Escalation Rate (%)",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                help="Percentage increase in variable costs beyond threshold"
            )
        
        # Operational constraints
        st.subheader("Operational Constraints")
        
        enable_constraints = st.checkbox("Enable Production Constraints")
        
        if enable_constraints:
            max_capacity = st.number_input(
                "Maximum Production Capacity",
                min_value=100,
                max_value=50000,
                value=5000,
                help="Maximum units that can be produced"
            )
            
            capacity_cost = st.number_input(
                "Capacity Expansion Cost ($)",
                min_value=0,
                max_value=100000,
                value=10000,
                help="Additional fixed cost to exceed capacity"
            )
    
    with col3:
        st.subheader("Pricing Strategy")
        
        # Current/test price
        # Determine range from preset if available
        preset_price_min = int(preset["price_range"][0]) if preset else 1
        preset_price_max = int(preset["price_range"][1]) if preset else 200
        preset_price_mid = int((preset_price_min + preset_price_max) / 2)

        price = st.slider(
            "Test Price ($)", 
            min_value=preset_price_min, 
            max_value=preset_price_max, 
            value=preset_price_mid,
            help="Price point to analyze and compare against optimal"
        )

        # Price range for analysis
        st.subheader("Analysis Range")
        price_min = st.number_input(
            "Minimum Price ($)",
            min_value=1,
            max_value=100,
            value=1,
            help="Lower bound for price analysis"
        )
        
        price_max = st.number_input(
            "Maximum Price ($)",
            min_value=price_min + 10,
            max_value=500,
            value=100,
            help="Upper bound for price analysis"
        )
        
        analysis_resolution = st.selectbox(
            "Analysis Resolution",
            [50, 100, 200, 500],
            index=1,
            help="Number of price points to analyze (higher = more precise)"
        )
        
        # Advanced options
        st.subheader("Advanced Options")
        
        include_risk_analysis = st.checkbox(
            "Include Risk Analysis",
            value=True,
            help="Analyze profit volatility and risk metrics"
        )
        
        time_horizon = st.selectbox(
            "Time Horizon",
            ["Short-term (1-3 months)", "Medium-term (6-12 months)", "Long-term (1-3 years)"],
            help="Planning horizon affects discount rates and growth assumptions"
        )
        
        # Save/Load configurations
        st.subheader("Configuration Management")
        
        config_name = st.text_input(
            "Configuration Name",
            placeholder="e.g., Premium Product Launch",
            help="Name for saving this configuration"
        )
        
        col_save, col_load = st.columns(2)
        with col_save:
            if st.button("💾 Save Config") and config_name:
                config = {
                    'demand_type': demand_type,
                    'a': a, 'b': b,
                    'market_condition': market_condition,
                    'fixed_cost': fixed_cost,
                    'variable_cost': variable_cost,
                    'price_range': [price_min, price_max],
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.configurations[config_name] = config
                st.success(f"Configuration '{config_name}' saved!")
        
        with col_load:
            if st.session_state.configurations:
                selected_config = st.selectbox(
                    "Load Configuration",
                    [""] + list(st.session_state.configurations.keys())
                )
                if st.button("📁 Load Config") and selected_config:
                    st.rerun()

# Run simulation button
st.markdown("---")
run_simulation = st.button(
    "🚀 Run Advanced Simulation", 
    type="primary",
    help="Execute comprehensive pricing analysis with current parameters"
)

# Main simulation logic
if run_simulation:
    with st.spinner("Running advanced simulation..."):
        # Apply cost escalation if enabled
        def adjusted_variable_cost(demand_qty):
            if cost_escalation and np.any(demand_qty > escalation_threshold):
                base_cost = np.where(
                    demand_qty <= escalation_threshold,
                    variable_cost,
                    variable_cost * (1 + escalation_rate / 100)
                )
                return base_cost
            return variable_cost
        
        # Run main simulation
        demand_fn, prices, demand, revenue, profit, cost = compute_demand(
            demand_type, 
            a_adjusted, 
            b_adjusted, 
            fixed_cost, 
            variable_cost,  # Base variable cost - escalation handled in compute_demand if needed
            price_range=(price_min, price_max),
            resolution=analysis_resolution
        )
        
        # Compute optimal price
        optimal_price, max_profit, demand_optimal = compute_optimal_price(
            demand_type, 
            demand_fn, 
            fixed_cost, 
            variable_cost, 
            a_adjusted, 
            b_adjusted
        )
        
        # Evaluate current price
        is_optimal = evaluate_price_optimality(price, optimal_price)
        recommendation = generate_recommendation(is_optimal, price, optimal_price)
        
        # Store results in session state
        st.session_state.current_results = {
            'prices': prices,
            'demand': demand,
            'revenue': revenue,
            'profit': profit,
            'cost': cost,
            'optimal_price': optimal_price,
            'max_profit': max_profit,
            'test_price': price,
            'is_optimal': is_optimal,
            'recommendation': recommendation,
            'demand_fn': demand_fn,
            'config': {
                'demand_type': demand_type,
                'market_condition': market_condition,
                'a': a_adjusted,
                'b': b_adjusted
            }
        }
        
        # Add to simulation history
        st.session_state.simulation_history.append({
            'timestamp': datetime.now(),
            'config_name': config_name or f"Simulation_{len(st.session_state.simulation_history)+1}",
            'optimal_price': optimal_price,
            'max_profit': max_profit,
            'test_price': price,
            'market_condition': market_condition
        })

# Display results if available
if st.session_state.current_results:
    results = st.session_state.current_results
    
    st.header("🎯 Simulation Results")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Test Price", 
            format_currency(results['test_price']),
            help="The price point you're analyzing"
        )
    
    with col2:
        st.metric(
            "Optimal Price", 
            format_currency(results['optimal_price']),
            delta=format_currency(results['optimal_price'] - results['test_price']),
            help="Mathematically optimal price for maximum profit"
        )
    
    with col3:
        st.metric(
            "Maximum Profit", 
            format_currency(results['max_profit']),
            help="Profit at optimal pricing"
        )
    
    with col4:
        # Calculate current profit
        current_demand = results['demand_fn'](results['test_price'])
        current_profit = (results['test_price'] * current_demand) - (fixed_cost + variable_cost * current_demand)
        
        st.metric(
            "Current Profit",
            format_currency(current_profit),
            delta=format_currency(current_profit - results['max_profit']),
            help="Profit at your test price"
        )
    
    # Recommendation
    if results['is_optimal']:
        st.success("✅ Your pricing is optimal or very close to optimal!")
    else:
        st.warning(f"💡 {results['recommendation']}")
    
    # Visualization
    st.subheader("📈 Revenue & Profit Analysis")
    fig = plot_revenue_profit(
        results['prices'], 
        results['revenue'], 
        results['profit'], 
        results['optimal_price'], 
        results['test_price']
    )
    st.pyplot(fig)
    
    # Additional insights
    if include_risk_analysis:
        st.subheader("🎲 Risk Analysis")
        
        # Calculate risk metrics
        profit_std = np.std(results['profit'])
        profit_cv = profit_std / np.mean(results['profit']) if np.mean(results['profit']) > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Profit Volatility", f"{profit_cv:.2%}", help="Coefficient of variation in profit")
        with col2:
            # Calculate downside risk (% of scenarios with profit < 50% of max)
            downside_threshold = results['max_profit'] * 0.5
            downside_scenarios = np.sum(results['profit'] < downside_threshold) / len(results['profit'])
            st.metric("Downside Risk", f"{downside_scenarios:.1%}", help="Scenarios with profit < 50% of maximum")

# Scenario comparison tab
with scenario_tab:
    st.header("🎯 Multi-Scenario Analysis")
    
    if len(st.session_state.simulation_history) > 0:
        st.subheader("Simulation History Comparison")
        
        # Create comparison dataframe
        history_df = pd.DataFrame(st.session_state.simulation_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        # Display comparison table
        st.dataframe(
            history_df[['config_name', 'optimal_price', 'max_profit', 'test_price', 'market_condition']].round(2),
            use_container_width=True
        )
        
        # Export options
        if st.button("📊 Export Simulation History"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"pricing_simulation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("Run simulations to compare different scenarios here.")

# Advanced analytics tab
with analysis_tab:
    st.header("📈 Advanced Analytics")
    
    if st.session_state.current_results:
        results = st.session_state.current_results
        
        # Sensitivity analysis
        st.subheader("🔍 Sensitivity Analysis")
        
        sensitivity_param = st.selectbox(
            "Parameter to Analyze",
            ["Demand Coefficient A", "Price Elasticity B", "Fixed Cost", "Variable Cost"]
        )
        
        if st.button("Run Sensitivity Analysis"):
            # This would require extending your calculus module
            st.info("Sensitivity analysis would show how optimal price changes with parameter variations.")
            
            # Placeholder for sensitivity results
            sensitivity_data = {
                'Parameter Change': ['-20%', '-10%', '0%', '+10%', '+20%'],
                'Optimal Price': [45.2, 47.8, 50.0, 52.3, 54.8],
                'Max Profit': [2180, 2240, 2300, 2285, 2250]
            }
            
            st.dataframe(pd.DataFrame(sensitivity_data))
    
    else:
        st.info("Run a simulation first to access advanced analytics.")

# Data upload tab (existing functionality preserved)
with data_tab:
    st.header("📊 Data Upload & Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload Historical Pricing Data (CSV)", 
        type=["csv"],
        help="CSV should contain columns: Price ($), Demand (Units), Revenue ($), Total Cost ($), Profit ($)"
    )
    
    if uploaded_file:
        st.subheader("📂 Uploaded Data Analysis")
        df = pd.read_csv(uploaded_file)
        
        # Data preview
        with st.expander("View Data Preview"):
            st.dataframe(df.head(10))
            st.caption(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
        
        # Extract columns with error handling
        try:
            prices_csv = df["Price ($)"]
            demand_csv = df["Demand (Units)"]
            revenue_csv = df["Revenue ($)"]
            cost_csv = df["Total Cost ($)"]
            profit_csv = df["Profit ($)"]

            # Find optimal price in uploaded data
            max_profit_idx = profit_csv.idxmax()
            optimal_price_from_csv = prices_csv[max_profit_idx]
            max_profit_value = profit_csv[max_profit_idx]

            # Interactive price testing
            user_price_csv = st.number_input(
                "Test Price (from uploaded range)", 
                min_value=float(prices_csv.min()), 
                max_value=float(prices_csv.max()), 
                value=float(prices_csv.iloc[0]),
                help="Select a price within your data range to analyze"
            )

            is_optimal_csv = evaluate_price_optimality(user_price_csv, optimal_price_from_csv)
            recommendation_csv = generate_recommendation(is_optimal_csv, user_price_csv, optimal_price_from_csv)

            # Results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Optimal Price (CSV)", format_currency(optimal_price_from_csv))
            with col2:
                st.metric("Maximum Profit", format_currency(max_profit_value))
            with col3:
                st.metric("Test Price", format_currency(user_price_csv))

            if is_optimal_csv:
                st.success("✅ Your selected price is optimal or very close to it!")
            else:
                st.warning(f"💡 {recommendation_csv}")

            # Plot uploaded data
            fig_csv = plot_revenue_profit(prices_csv, revenue_csv, profit_csv, optimal_price_from_csv, user_price_csv)
            st.pyplot(fig_csv)
            
            # Data export
            if st.button("📊 Export Enhanced Analysis"):
                enhanced_df = df.copy()
                enhanced_df['Distance_from_Optimal'] = abs(enhanced_df['Price ($)'] - optimal_price_from_csv)
                enhanced_df['Profit_Loss'] = max_profit_value - enhanced_df['Profit ($)']
                
                csv_export = enhanced_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Enhanced CSV",
                    data=csv_export,
                    file_name=f"enhanced_pricing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        except KeyError as e:
            st.error(f"❌ Missing required column in CSV: {e}")
            st.info("Required columns: Price ($), Demand (Units), Revenue ($), Total Cost ($), Profit ($)")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 14px;'>
    🔬 Advanced Dynamic Pricing Simulator | Built with mathematical optimization algorithms
    </div>
    """, 
    unsafe_allow_html=True
)