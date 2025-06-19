import numpy as np

def linear_demand(p, a, b):
    return np.maximum(a - b * p, 0)

def exponential_demand(p, a, b):
    return np.maximum(a * np.exp(-b * p), 0)

def compute_demand(demand_type, a, b, fixed_cost, variable_cost, price_range=(1, 100), resolution=100):
    prices = np.linspace(price_range[0], price_range[1], resolution)

    if demand_type == "Linear":
        demand_fn = lambda p: linear_demand(p, a, b)
    elif demand_type == "Exponential":
        demand_fn = lambda p: exponential_demand(p, a, b)
    else:
        raise ValueError("Unsupported demand type")

    demand = demand_fn(prices)
    revenue = prices * demand
    cost = fixed_cost + variable_cost * demand
    profit = revenue - cost

    return demand_fn, prices, demand, revenue, profit, cost
