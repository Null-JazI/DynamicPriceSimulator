#demand.py
"""
Enhanced demand computation module with support for advanced cost models,
capacity constraints, and market condition adjustments.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any
import warnings

def linear_demand(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Linear demand function: D(p) = max(a - b*p, 0)
    
    Args:
        p: Price array
        a: Maximum demand (intercept)
        b: Price elasticity (slope)
    
    Returns:
        Demand quantities (non-negative)
    """
    return np.maximum(a - b * p, 0)

def exponential_demand(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Exponential demand function: D(p) = max(a * exp(-b*p), 0)
    
    Args:
        p: Price array
        a: Base demand level
        b: Decay rate
    
    Returns:
        Demand quantities (non-negative)
    """
    return np.maximum(a * np.exp(-b * p), 0)

def power_demand(p: np.ndarray, a: float, b: float, min_price: float = 0.01) -> np.ndarray:
    """
    Power demand function: D(p) = a * p^(-b) for p >= min_price
    
    Args:
        p: Price array
        a: Scale parameter
        b: Elasticity exponent (should be positive)
        min_price: Minimum price to avoid division issues
    
    Returns:
        Demand quantities
    """
    # Ensure prices are above minimum to avoid numerical issues
    p_safe = np.maximum(p, min_price)
    return a * np.power(p_safe, -b)

def logistic_demand(p: np.ndarray, a: float, b: float, c: float = 50.0) -> np.ndarray:
    """
    Logistic demand function: D(p) = a / (1 + exp(b*(p-c)))
    
    Args:
        p: Price array
        a: Maximum demand asymptote
        b: Steepness parameter
        c: Inflection point (price where demand = a/2)
    
    Returns:
        Demand quantities
    """
    return a / (1 + np.exp(b * (p - c)))

def compute_variable_cost_with_escalation(
    demand: np.ndarray, 
    base_variable_cost: float,
    escalation_threshold: float = 1000,
    escalation_rate: float = 10.0
) -> np.ndarray:
    """
    Compute variable costs with escalation beyond threshold
    
    Args:
        demand: Demand quantities
        base_variable_cost: Base per-unit variable cost
        escalation_threshold: Volume threshold for cost escalation
        escalation_rate: Percentage increase beyond threshold
    
    Returns:
        Array of effective variable costs per unit
    """
    escalation_multiplier = 1 + (escalation_rate / 100.0)
    
    # Apply escalation where demand exceeds threshold
    effective_cost = np.where(
        demand > escalation_threshold,
        base_variable_cost * escalation_multiplier,
        base_variable_cost
    )
    
    return effective_cost

def apply_capacity_constraints(
    demand: np.ndarray,
    prices: np.ndarray,
    max_capacity: int,
    capacity_expansion_cost: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply production capacity constraints to demand
    
    Args:
        demand: Unconstrained demand
        prices: Price array
        max_capacity: Maximum production capacity
        capacity_expansion_cost: Additional fixed cost to exceed capacity
    
    Returns:
        Tuple of (constrained_demand, additional_fixed_costs)
    """
    constrained_demand = np.minimum(demand, max_capacity)
    
    # Calculate additional costs when demand exceeds capacity
    exceeds_capacity = demand > max_capacity
    additional_costs = np.where(exceeds_capacity, capacity_expansion_cost, 0)
    
    return constrained_demand, additional_costs

def compute_advanced_costs(
    demand: np.ndarray,
    base_fixed_cost: float,
    base_variable_cost: float,
    enable_escalation: bool = False,
    escalation_threshold: float = 1000,
    escalation_rate: float = 10.0,
    enable_constraints: bool = False,
    max_capacity: int = 5000,
    capacity_cost: float = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute advanced cost structure with escalation and constraints
    
    Returns:
        Tuple of (total_costs, fixed_costs, variable_costs_per_unit)
    """
    # Base variable cost per unit
    if enable_escalation:
        var_cost_per_unit = compute_variable_cost_with_escalation(
            demand, base_variable_cost, escalation_threshold, escalation_rate
        )
    else:
        var_cost_per_unit = np.full_like(demand, base_variable_cost)
    
    # Apply capacity constraints if enabled
    if enable_constraints:
        constrained_demand, additional_fixed_costs = apply_capacity_constraints(
            demand, None, max_capacity, capacity_cost
        )
        total_fixed_costs = base_fixed_cost + additional_fixed_costs
        # Use constrained demand for variable cost calculation
        total_variable_costs = var_cost_per_unit * constrained_demand
    else:
        constrained_demand = demand
        total_fixed_costs = np.full_like(demand, base_fixed_cost)
        total_variable_costs = var_cost_per_unit * demand
    
    total_costs = total_fixed_costs + total_variable_costs
    
    return total_costs, total_fixed_costs, var_cost_per_unit

def apply_market_condition_modifiers(
    demand_base: np.ndarray,
    market_condition: str = "Normal"
) -> np.ndarray:
    """
    Apply market condition modifiers to base demand
    
    Args:
        demand_base: Base demand without market adjustments
        market_condition: Market condition type
    
    Returns:
        Adjusted demand
    """
    modifiers = {
        "Normal": 1.0,
        "High Competition": 0.8,
        "Premium Market": 0.6,
        "Economic Downturn": 0.7,
        "Market Growth": 1.3,
        "Recession": 0.5
    }
    
    multiplier = modifiers.get(market_condition, 1.0)
    return demand_base * multiplier

def compute_demand(
    demand_type: str,
    a: float,
    b: float,
    fixed_cost: float,
    variable_cost: float,
    price_range: Tuple[float, float] = (1, 100),
    resolution: int = 100,
    market_condition: str = "Normal",
    enable_cost_escalation: bool = False,
    escalation_threshold: float = 1000,
    escalation_rate: float = 10.0,
    enable_capacity_constraints: bool = False,
    max_capacity: int = 5000,
    capacity_cost: float = 10000,
    additional_params: Optional[Dict[str, Any]] = None
) -> Tuple[Callable, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Enhanced demand computation with advanced features
    
    Args:
        demand_type: Type of demand function ("Linear", "Exponential", "Power", "Logistic")
        a, b: Demand function parameters
        fixed_cost, variable_cost: Base cost parameters
        price_range: (min_price, max_price) for analysis
        resolution: Number of price points to analyze
        market_condition: Market condition modifier
        enable_cost_escalation: Enable variable cost escalation
        escalation_threshold: Volume threshold for cost escalation
        escalation_rate: Percentage increase in costs beyond threshold
        enable_capacity_constraints: Enable production capacity limits
        max_capacity: Maximum production capacity
        capacity_cost: Cost to exceed capacity
        additional_params: Additional parameters for advanced demand functions
    
    Returns:
        Tuple of (demand_function, prices, demand, revenue, profit, cost)
    """
    
    # Validate inputs
    if price_range[0] <= 0 or price_range[1] <= price_range[0]:
        raise ValueError("Invalid price range")
    
    if resolution < 10:
        raise ValueError("Resolution must be at least 10")
    
    # Generate price array
    prices = np.linspace(price_range[0], price_range[1], resolution)
    
    # Select and create demand function
    if demand_type == "Linear":
        demand_fn = lambda p: linear_demand(p, a, b)
        base_demand = linear_demand(prices, a, b)
        
    elif demand_type == "Exponential":
        demand_fn = lambda p: exponential_demand(p, a, b)
        base_demand = exponential_demand(prices, a, b)
        
    elif demand_type == "Power":
        # For power demand, need additional validation
        if b <= 0:
            warnings.warn("Power demand elasticity should be positive, using absolute value")
            b = abs(b)
        demand_fn = lambda p: power_demand(p, a, b)
        base_demand = power_demand(prices, a, b)
        
    elif demand_type == "Logistic":
        # For logistic demand, use additional parameter c if provided
        c = additional_params.get('c', price_range[1] * 0.5) if additional_params else price_range[1] * 0.5
        demand_fn = lambda p: logistic_demand(p, a, b, c)
        base_demand = logistic_demand(prices, a, b, c)
        
    else:
        raise ValueError(f"Unsupported demand type: {demand_type}")
    
    # Apply market condition modifiers
    adjusted_demand = apply_market_condition_modifiers(base_demand, market_condition)
    
    # Apply capacity constraints if enabled
    if enable_capacity_constraints:
        final_demand, _ = apply_capacity_constraints(
            adjusted_demand, prices, max_capacity, capacity_cost
        )
    else:
        final_demand = adjusted_demand
    
    # Compute costs with advanced features
    total_costs, fixed_costs, var_cost_per_unit = compute_advanced_costs(
        final_demand,
        fixed_cost,
        variable_cost,
        enable_cost_escalation,
        escalation_threshold,
        escalation_rate,
        enable_capacity_constraints,
        max_capacity,
        capacity_cost
    )
    
    # Compute revenue and profit
    revenue = prices * final_demand
    profit = revenue - total_costs
    
    # Return enhanced demand function that includes all modifiers
    def enhanced_demand_fn(p):
        if isinstance(p, (int, float)):
            p = np.array([p])
        
        # Apply base demand function
        if demand_type == "Linear":
            base_d = linear_demand(p, a, b)
        elif demand_type == "Exponential":
            base_d = exponential_demand(p, a, b)
        elif demand_type == "Power":
            base_d = power_demand(p, a, b)
        elif demand_type == "Logistic":
            c = additional_params.get('c', price_range[1] * 0.5) if additional_params else price_range[1] * 0.5
            base_d = logistic_demand(p, a, b, c)
        
        # Apply market condition
        adjusted_d = apply_market_condition_modifiers(base_d, market_condition)
        
        # Apply capacity constraints
        if enable_capacity_constraints:
            final_d, _ = apply_capacity_constraints(adjusted_d, p, max_capacity, capacity_cost)
        else:
            final_d = adjusted_d
        
        return final_d[0] if len(final_d) == 1 else final_d
    
    return enhanced_demand_fn, prices, final_demand, revenue, profit, total_costs

def compute_demand_elasticity(
    demand_fn: Callable,
    price: float,
    delta: float = 0.01
) -> float:
    """
    Compute price elasticity of demand at a given price point
    
    Args:
        demand_fn: Demand function
        price: Price point for elasticity calculation
        delta: Small change in price for numerical differentiation
    
    Returns:
        Price elasticity (negative value indicates normal good)
    """
    if price <= delta:
        return float('nan')
    
    q0 = demand_fn(price)
    q1 = demand_fn(price + delta)
    
    if q0 == 0:
        return float('inf') if q1 == 0 else float('-inf')
    
    elasticity = ((q1 - q0) / q0) / ((delta) / price)
    return elasticity

def compute_revenue_elasticity(
    demand_fn: Callable,
    price: float,
    delta: float = 0.01
) -> float:
    """
    Compute revenue elasticity at a given price point
    
    Args:
        demand_fn: Demand function
        price: Price point
        delta: Small change in price
    
    Returns:
        Revenue elasticity
    """
    r0 = price * demand_fn(price)
    r1 = (price + delta) * demand_fn(price + delta)
    
    if r0 == 0:
        return float('inf') if r1 == 0 else float('-inf')
    
    revenue_elasticity = ((r1 - r0) / r0) / (delta / price)
    return revenue_elasticity

def validate_demand_parameters(demand_type: str, a: float, b: float) -> Tuple[bool, str]:
    """
    Validate demand function parameters
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if demand_type == "Linear":
        if a <= 0:
            return False, "Linear demand parameter 'a' must be positive"
        if b <= 0:
            return False, "Linear demand parameter 'b' must be positive"
    
    elif demand_type == "Exponential":
        if a <= 0:
            return False, "Exponential demand parameter 'a' must be positive"
        if b <= 0:
            return False, "Exponential demand parameter 'b' must be positive"
    
    elif demand_type == "Power":
        if a <= 0:
            return False, "Power demand parameter 'a' must be positive"
        # b can be any positive value for power functions
    
    elif demand_type == "Logistic":
        if a <= 0:
            return False, "Logistic demand parameter 'a' must be positive"
        if b <= 0:
            return False, "Logistic demand parameter 'b' must be positive"
    
    else:
        return False, f"Unknown demand type: {demand_type}"
    
    return True, ""

if __name__ == "__main__":
    # Example usage and testing
    print("Testing enhanced demand computation...")
    
    # Test linear demand with cost escalation
    demand_fn, prices, demand, revenue, profit, cost = compute_demand(
        demand_type="Linear",
        a=1000,
        b=15.0,
        fixed_cost=2000,
        variable_cost=5.0,
        price_range=(1, 100),
        resolution=100,
        enable_cost_escalation=True,
        escalation_threshold=500,
        escalation_rate=20.0
    )
    
    print(f"Max profit: ${max(profit):.2f}")
    print(f"Optimal price around: ${prices[np.argmax(profit)]:.2f}")
    
    # Test parameter validation
    is_valid, error = validate_demand_parameters("Linear", 1000, 15.0)
    print(f"Parameters valid: {is_valid}")
    
    print("Enhanced demand module test completed!")