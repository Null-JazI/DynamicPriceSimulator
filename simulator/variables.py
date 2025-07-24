#variables.py
"""
Configuration management for Dynamic Pricing Simulator
Provides default values, validation, and configuration presets
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum

class DemandType(Enum):
    LINEAR = "Linear"
    EXPONENTIAL = "Exponential"

class MarketCondition(Enum):
    NORMAL = "Normal"
    HIGH_COMPETITION = "High Competition"
    PREMIUM_MARKET = "Premium Market"
    ECONOMIC_DOWNTURN = "Economic Downturn"

class TimeHorizon(Enum):
    SHORT_TERM = "Short-term (1-3 months)"
    MEDIUM_TERM = "Medium-term (6-12 months)"
    LONG_TERM = "Long-term (1-3 years)"

@dataclass
class DemandParameters:
    """Parameters for demand functions"""
    a: float  # Demand coefficient (market size for linear, base level for exponential)
    b: float  # Elasticity coefficient (slope for linear, decay rate for exponential)
    demand_type: DemandType
    
    def validate(self) -> bool:
        """Validate parameter ranges"""
        if self.demand_type == DemandType.LINEAR:
            return self.a >= 100 and self.b >= 0.1
        elif self.demand_type == DemandType.EXPONENTIAL:
            return self.a >= 100 and 0.01 <= self.b <= 1.0
        return False

@dataclass
class CostParameters:
    """Cost structure parameters"""
    fixed_cost: float
    variable_cost: float
    enable_escalation: bool = False
    escalation_threshold: float = 1000
    escalation_rate: float = 10.0  # percentage
    
    def validate(self) -> bool:
        """Validate cost parameters"""
        return (self.fixed_cost >= 0 and 
                self.variable_cost >= 0 and
                self.escalation_threshold > 0 and
                0 <= self.escalation_rate <= 100)

@dataclass
class OperationalConstraints:
    """Operational and capacity constraints"""
    enable_constraints: bool = False
    max_capacity: int = 5000
    capacity_cost: float = 10000
    
    def validate(self) -> bool:
        """Validate operational constraints"""
        return (self.max_capacity > 0 and 
                self.capacity_cost >= 0)

@dataclass
class AnalysisParameters:
    """Analysis and simulation parameters"""
    price_min: float = 1.0
    price_max: float = 100.0
    resolution: int = 100
    include_risk_analysis: bool = True
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    
    def validate(self) -> bool:
        """Validate analysis parameters"""
        return (self.price_min > 0 and 
                self.price_max > self.price_min and
                self.resolution >= 50)

@dataclass
class SimulationConfig:
    """Complete simulation configuration"""
    demand_params: DemandParameters
    cost_params: CostParameters
    operational_constraints: OperationalConstraints
    analysis_params: AnalysisParameters
    market_condition: MarketCondition = MarketCondition.NORMAL
    
    def validate(self) -> bool:
        """Validate entire configuration"""
        return (self.demand_params.validate() and
                self.cost_params.validate() and
                self.operational_constraints.validate() and
                self.analysis_params.validate())

# Default configurations
DEFAULT_CONFIGS = {
    "basic_product": SimulationConfig(
        demand_params=DemandParameters(
            a=1000,
            b=15.0,
            demand_type=DemandType.LINEAR
        ),
        cost_params=CostParameters(
            fixed_cost=2000,
            variable_cost=5.0
        ),
        operational_constraints=OperationalConstraints(),
        analysis_params=AnalysisParameters()
    ),
    
    "premium_product": SimulationConfig(
        demand_params=DemandParameters(
            a=500,
            b=8.0,
            demand_type=DemandType.LINEAR
        ),
        cost_params=CostParameters(
            fixed_cost=5000,
            variable_cost=15.0
        ),
        operational_constraints=OperationalConstraints(),
        analysis_params=AnalysisParameters(
            price_min=20.0,
            price_max=200.0
        ),
        market_condition=MarketCondition.PREMIUM_MARKET
    ),
    
    "competitive_market": SimulationConfig(
        demand_params=DemandParameters(
            a=2000,
            b=25.0,
            demand_type=DemandType.LINEAR
        ),
        cost_params=CostParameters(
            fixed_cost=1500,
            variable_cost=3.0
        ),
        operational_constraints=OperationalConstraints(),
        analysis_params=AnalysisParameters(),
        market_condition=MarketCondition.HIGH_COMPETITION
    ),
    
    "digital_product": SimulationConfig(
        demand_params=DemandParameters(
            a=5000,
            b=0.05,
            demand_type=DemandType.EXPONENTIAL
        ),
        cost_params=CostParameters(
            fixed_cost=10000,
            variable_cost=0.5
        ),
        operational_constraints=OperationalConstraints(),
        analysis_params=AnalysisParameters(
            price_min=5.0,
            price_max=50.0
        )
    )
}

# Market condition modifiers
MARKET_CONDITION_MODIFIERS = {
    MarketCondition.NORMAL: {
        "demand_multiplier": 1.0,
        "elasticity_multiplier": 1.0,
        "cost_multiplier": 1.0
    },
    MarketCondition.HIGH_COMPETITION: {
        "demand_multiplier": 0.8,
        "elasticity_multiplier": 1.5,
        "cost_multiplier": 0.95
    },
    MarketCondition.PREMIUM_MARKET: {
        "demand_multiplier": 0.6,
        "elasticity_multiplier": 0.7,
        "cost_multiplier": 1.1
    },
    MarketCondition.ECONOMIC_DOWNTURN: {
        "demand_multiplier": 0.7,
        "elasticity_multiplier": 1.8,
        "cost_multiplier": 1.05
    }
}

# Time horizon adjustments
TIME_HORIZON_ADJUSTMENTS = {
    TimeHorizon.SHORT_TERM: {
        "discount_rate": 0.02,
        "uncertainty_factor": 1.0,
        "growth_assumption": 0.0
    },
    TimeHorizon.MEDIUM_TERM: {
        "discount_rate": 0.05,
        "uncertainty_factor": 1.2,
        "growth_assumption": 0.03
    },
    TimeHorizon.LONG_TERM: {
        "discount_rate": 0.08,
        "uncertainty_factor": 1.5,
        "growth_assumption": 0.05
    }
}

def get_default_config(product_type: str = "basic_product") -> SimulationConfig:
    """Get a default configuration for a product type"""
    if product_type in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[product_type]
    return DEFAULT_CONFIGS["basic_product"]

def apply_market_condition_modifiers(
    demand_params: DemandParameters, 
    cost_params: CostParameters,
    market_condition: MarketCondition
) -> Tuple[DemandParameters, CostParameters]:
    """Apply market condition modifiers to parameters"""
    
    modifiers = MARKET_CONDITION_MODIFIERS[market_condition]
    
    # Modify demand parameters
    modified_demand = DemandParameters(
        a=demand_params.a * modifiers["demand_multiplier"],
        b=demand_params.b * modifiers["elasticity_multiplier"],
        demand_type=demand_params.demand_type
    )
    
    # Modify cost parameters
    modified_cost = CostParameters(
        fixed_cost=cost_params.fixed_cost * modifiers["cost_multiplier"],
        variable_cost=cost_params.variable_cost * modifiers["cost_multiplier"],
        enable_escalation=cost_params.enable_escalation,
        escalation_threshold=cost_params.escalation_threshold,
        escalation_rate=cost_params.escalation_rate
    )
    
    return modified_demand, modified_cost

def validate_price_range(price_min: float, price_max: float, variable_cost: float) -> bool:
    """Validate that price range makes economic sense"""
    return (price_min > 0 and 
            price_max > price_min and 
            price_min > variable_cost * 0.5)  # Minimum price should cover at least 50% of variable cost

def get_suggested_price_range(
    demand_params: DemandParameters, 
    cost_params: CostParameters
) -> Tuple[float, float]:
    """Suggest reasonable price range based on parameters"""
    
    # Base suggestion on cost structure and demand type
    min_price = max(1.0, cost_params.variable_cost * 1.2)  # At least 20% markup
    
    if demand_params.demand_type == DemandType.LINEAR:
        # For linear demand: find price where demand approaches zero
        max_reasonable_price = min(200.0, demand_params.a / demand_params.b * 0.8)
    else:  # Exponential
        # For exponential demand: use multiple of variable cost
        max_reasonable_price = min(100.0, cost_params.variable_cost * 15)
    
    return min_price, max_reasonable_price

def calculate_break_even_price(cost_params: CostParameters, expected_volume: int) -> float:
    """Calculate break-even price for given expected volume"""
    if expected_volume <= 0:
        return float('inf')
    
    return (cost_params.fixed_cost / expected_volume) + cost_params.variable_cost

def get_risk_tolerance_adjustments(risk_level: str) -> Dict[str, float]:
    """Get adjustments based on risk tolerance"""
    adjustments = {
        "conservative": {
            "price_buffer": 0.95,  # Price 5% below optimal
            "demand_safety_margin": 0.9,  # Assume 10% lower demand
            "cost_buffer": 1.1  # Assume 10% higher costs
        },
        "moderate": {
            "price_buffer": 0.98,
            "demand_safety_margin": 0.95,
            "cost_buffer": 1.05
        },
        "aggressive": {
            "price_buffer": 1.02,  # Price 2% above optimal
            "demand_safety_margin": 1.05,  # Assume 5% higher demand
            "cost_buffer": 0.98  # Assume 2% lower costs
        }
    }
    
    return adjustments.get(risk_level, adjustments["moderate"])

def generate_scenario_matrix(base_config: SimulationConfig) -> List[Dict[str, Any]]:
    """Generate multiple scenarios for comparison"""
    
    scenarios = []
    base_a = base_config.demand_params.a
    base_b = base_config.demand_params.b
    base_fixed = base_config.cost_params.fixed_cost
    base_variable = base_config.cost_params.variable_cost
    
    # Scenario variations
    variations = [
        {"name": "Pessimistic", "demand_mult": 0.8, "elasticity_mult": 1.3, "cost_mult": 1.2},
        {"name": "Realistic", "demand_mult": 1.0, "elasticity_mult": 1.0, "cost_mult": 1.0},
        {"name": "Optimistic", "demand_mult": 1.2, "elasticity_mult": 0.8, "cost_mult": 0.9}
    ]
    
    for var in variations:
        scenario = {
            "name": var["name"],
            "demand_params": DemandParameters(
                a=base_a * var["demand_mult"],
                b=base_b * var["elasticity_mult"],
                demand_type=base_config.demand_params.demand_type
            ),
            "cost_params": CostParameters(
                fixed_cost=base_fixed * var["cost_mult"],
                variable_cost=base_variable * var["cost_mult"],
                enable_escalation=base_config.cost_params.enable_escalation,
                escalation_threshold=base_config.cost_params.escalation_threshold,
                escalation_rate=base_config.cost_params.escalation_rate
            )
        }
        scenarios.append(scenario)
    
    return scenarios

def export_configuration(config: SimulationConfig, filename: str = None) -> Dict[str, Any]:
    """Export configuration to dictionary for JSON serialization"""
    
    config_dict = {
        "demand_parameters": {
            "a": config.demand_params.a,
            "b": config.demand_params.b,
            "demand_type": config.demand_params.demand_type.value
        },
        "cost_parameters": {
            "fixed_cost": config.cost_params.fixed_cost,
            "variable_cost": config.cost_params.variable_cost,
            "enable_escalation": config.cost_params.enable_escalation,
            "escalation_threshold": config.cost_params.escalation_threshold,
            "escalation_rate": config.cost_params.escalation_rate
        },
        "operational_constraints": {
            "enable_constraints": config.operational_constraints.enable_constraints,
            "max_capacity": config.operational_constraints.max_capacity,
            "capacity_cost": config.operational_constraints.capacity_cost
        },
        "analysis_parameters": {
            "price_min": config.analysis_params.price_min,
            "price_max": config.analysis_params.price_max,
            "resolution": config.analysis_params.resolution,
            "include_risk_analysis": config.analysis_params.include_risk_analysis,
            "time_horizon": config.analysis_params.time_horizon.value
        },
        "market_condition": config.market_condition.value
    }
    
    return config_dict

def import_configuration(config_dict: Dict[str, Any]) -> SimulationConfig:
    """Import configuration from dictionary"""
    
    return SimulationConfig(
        demand_params=DemandParameters(
            a=config_dict["demand_parameters"]["a"],
            b=config_dict["demand_parameters"]["b"],
            demand_type=DemandType(config_dict["demand_parameters"]["demand_type"])
        ),
        cost_params=CostParameters(
            fixed_cost=config_dict["cost_parameters"]["fixed_cost"],
            variable_cost=config_dict["cost_parameters"]["variable_cost"],
            enable_escalation=config_dict["cost_parameters"].get("enable_escalation", False),
            escalation_threshold=config_dict["cost_parameters"].get("escalation_threshold", 1000),
            escalation_rate=config_dict["cost_parameters"].get("escalation_rate", 10.0)
        ),
        operational_constraints=OperationalConstraints(
            enable_constraints=config_dict["operational_constraints"].get("enable_constraints", False),
            max_capacity=config_dict["operational_constraints"].get("max_capacity", 5000),
            capacity_cost=config_dict["operational_constraints"].get("capacity_cost", 10000)
        ),
        analysis_params=AnalysisParameters(
            price_min=config_dict["analysis_parameters"]["price_min"],
            price_max=config_dict["analysis_parameters"]["price_max"],
            resolution=config_dict["analysis_parameters"]["resolution"],
            include_risk_analysis=config_dict["analysis_parameters"].get("include_risk_analysis", True),
            time_horizon=TimeHorizon(config_dict["analysis_parameters"].get("time_horizon", "Medium-term (6-12 months)"))
        ),
        market_condition=MarketCondition(config_dict.get("market_condition", "Normal"))
    )

# Industry-specific presets
INDUSTRY_PRESETS = {
    "retail_fashion": {
        "name": "Retail Fashion",
        "demand_type": DemandType.LINEAR,
        "a": 800, "b": 12.0,
        "fixed_cost": 3000, "variable_cost": 8.0,
        "price_range": (15, 80),
        "market_condition": MarketCondition.HIGH_COMPETITION
    },
    
    "software_saas": {
        "name": "Software (SaaS)",
        "demand_type": DemandType.EXPONENTIAL,
        "a": 2000, "b": 0.08,
        "fixed_cost": 15000, "variable_cost": 2.0,
        "price_range": (10, 100),
        "market_condition": MarketCondition.NORMAL
    },
    
    "luxury_goods": {
        "name": "Luxury Goods",
        "demand_type": DemandType.LINEAR,
        "a": 200, "b": 2.0,
        "fixed_cost": 10000, "variable_cost": 50.0,
        "price_range": (100, 1000),
        "market_condition": MarketCondition.PREMIUM_MARKET
    },
    
    "commodity": {
        "name": "Commodity Product",
        "demand_type": DemandType.LINEAR,
        "a": 5000, "b": 80.0,
        "fixed_cost": 5000, "variable_cost": 2.0,
        "price_range": (3, 20),
        "market_condition": MarketCondition.HIGH_COMPETITION
    },
    
    "consulting_services": {
        "name": "Consulting Services",
        "demand_type": DemandType.EXPONENTIAL,
        "a": 100, "b": 0.02,
        "fixed_cost": 2000, "variable_cost": 25.0,
        "price_range": (50, 500),
        "market_condition": MarketCondition.PREMIUM_MARKET
    }
}

def get_industry_preset(industry: str) -> SimulationConfig:
    """Get a preset configuration for specific industry"""
    
    if industry not in INDUSTRY_PRESETS:
        return get_default_config()
    
    preset = INDUSTRY_PRESETS[industry]
    
    return SimulationConfig(
        demand_params=DemandParameters(
            a=preset["a"],
            b=preset["b"],
            demand_type=preset["demand_type"]
        ),
        cost_params=CostParameters(
            fixed_cost=preset["fixed_cost"],
            variable_cost=preset["variable_cost"]
        ),
        operational_constraints=OperationalConstraints(),
        analysis_params=AnalysisParameters(
            price_min=preset["price_range"][0],
            price_max=preset["price_range"][1]
        ),
        market_condition=preset["market_condition"]
    )

# Parameter bounds for validation
PARAMETER_BOUNDS = {
    "linear_a": {"min": 100, "max": 100000, "default": 1000},
    "linear_b": {"min": 0.1, "max": 1000, "default": 15.0},
    "exponential_a": {"min": 100, "max": 50000, "default": 1000},
    "exponential_b": {"min": 0.001, "max": 1.0, "default": 0.1},
    "fixed_cost": {"min": 0, "max": 10000000, "default": 2000},
    "variable_cost": {"min": 0, "max": 10000, "default": 5.0},
    "price_min": {"min": 0.01, "max": 1000, "default": 1.0},
    "price_max": {"min": 1, "max": 10000, "default": 100.0},
    "resolution": {"min": 10, "max": 1000, "default": 100}
}

def validate_parameter(param_name: str, value: float) -> Tuple[bool, str]:
    """Validate a single parameter against bounds"""
    
    if param_name not in PARAMETER_BOUNDS:
        return True, ""
    
    bounds = PARAMETER_BOUNDS[param_name]
    
    if value < bounds["min"]:
        return False, f"{param_name} must be at least {bounds['min']}"
    
    if value > bounds["max"]:
        return False, f"{param_name} must be at most {bounds['max']}"
    
    return True, ""

def get_parameter_suggestions(demand_type: DemandType, industry: str = None) -> Dict[str, float]:
    """Get parameter suggestions based on demand type and industry"""
    
    if industry and industry in INDUSTRY_PRESETS:
        preset = INDUSTRY_PRESETS[industry]
        return {
            "a": preset["a"],
            "b": preset["b"],
            "fixed_cost": preset["fixed_cost"],
            "variable_cost": preset["variable_cost"],
            "price_min": preset["price_range"][0],
            "price_max": preset["price_range"][1]
        }
    
    # Default suggestions based on demand type
    if demand_type == DemandType.LINEAR:
        return {
            "a": PARAMETER_BOUNDS["linear_a"]["default"],
            "b": PARAMETER_BOUNDS["linear_b"]["default"],
            "fixed_cost": PARAMETER_BOUNDS["fixed_cost"]["default"],
            "variable_cost": PARAMETER_BOUNDS["variable_cost"]["default"],
            "price_min": PARAMETER_BOUNDS["price_min"]["default"],
            "price_max": PARAMETER_BOUNDS["price_max"]["default"]
        }
    else:  # Exponential
        return {
            "a": PARAMETER_BOUNDS["exponential_a"]["default"],
            "b": PARAMETER_BOUNDS["exponential_b"]["default"],
            "fixed_cost": PARAMETER_BOUNDS["fixed_cost"]["default"],
            "variable_cost": PARAMETER_BOUNDS["variable_cost"]["default"],
            "price_min": PARAMETER_BOUNDS["price_min"]["default"],
            "price_max": PARAMETER_BOUNDS["price_max"]["default"]
        }

if __name__ == "__main__":
    # Example usage and testing
    config = get_default_config("premium_product")
    print(f"Default premium product config: {config.demand_params.a}, {config.demand_params.b}")
    
    # Test validation
    print(f"Config is valid: {config.validate()}")
    
    # Test industry preset
    luxury_config = get_industry_preset("luxury_goods")
    print(f"Luxury goods config: {luxury_config.demand_params.a}, {luxury_config.cost_params.variable_cost}")