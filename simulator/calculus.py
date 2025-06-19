import sympy as sp
import numpy as np
from scipy.optimize import minimize_scalar

# ---------- LINEAR: Symbolic Optimization ----------
def compute_optimal_price_linear(fixed_cost, variable_cost, a, b):
    p = sp.Symbol('p', real=True, positive=True)
    D = a - b * p

    R = p * D
    C = fixed_cost + variable_cost * D
    Pi = R - C

    dPi = sp.diff(Pi, p)
    d2Pi = sp.diff(dPi, p)
    critical_points = sp.solve(dPi, p)

    for point in critical_points:
        if point.is_real and d2Pi.subs(p, point) < 0:
            optimal_price = float(point)
            demand_val = float(D.subs(p, optimal_price))
            revenue = optimal_price * demand_val
            cost = fixed_cost + variable_cost * demand_val
            profit = revenue - cost
            return optimal_price, profit

    return None, None

# ---------- EXPONENTIAL: Numerical Optimization ----------
def compute_optimal_price_exponential(demand_fn, fixed_cost, variable_cost, price_bounds=(1, 100)):
    def profit(p):
        d = demand_fn(p)
        r = p * d
        c = fixed_cost + variable_cost * d
        return -(r - c)

    result = minimize_scalar(profit, bounds=price_bounds, method='bounded')

    if result.success:
        optimal_price = result.x
        d = demand_fn(optimal_price)
        r = optimal_price * d
        c = fixed_cost + variable_cost * d
        return optimal_price, r - c
    else:
        return None, None

# ---------- Master Function ----------
def compute_optimal_price(demand_type, demand_fn, fixed_cost, variable_cost, a=1000, b=15):
    if demand_type == "Linear":
        return compute_optimal_price_linear(fixed_cost, variable_cost, a, b)
    elif demand_type == "Exponential":
        return compute_optimal_price_exponential(demand_fn, fixed_cost, variable_cost)
    else:
        raise ValueError("Unsupported demand type")
