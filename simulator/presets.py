# Industry Presets for Dynamic Pricing Simulator
# Each preset includes realistic demand models and cost structures

INDUSTRY_PRESETS = {
    "luxury_goods": {
        "industry_name": "Luxury Goods",
        "description": "Premium products where higher prices often signal quality and exclusivity. Exhibits Veblen effect - demand may increase with price in certain ranges.",
        "demand_type": "Exponential",
        "a": 50,  # Base demand intercept
        "b": 0.8,  # Price sensitivity (less elastic than normal goods)
        "fixed_cost": 100000,  # High fixed costs for brand building, R&D
        "variable_cost": 200,   # High-quality materials and craftsmanship
        "price_range": [500, 5000],
        "rationale": "Luxury goods have inelastic demand with exponential decay. High fixed costs reflect brand investment and marketing. Variable costs reflect premium materials and craftsmanship."
    },
    
    "tech_gadgets": {
        "industry_name": "Consumer Tech Gadgets",
        "description": "Technology products with network effects and early adopter dynamics. Price sensitivity varies by product lifecycle stage.",
        "demand_type": "Linear",
        "a": 1000,  # Strong initial demand for new tech
        "b": 2.5,   # Moderate price sensitivity
        "fixed_cost": 500000,  # High R&D and manufacturing setup
        "variable_cost": 150,   # Components and assembly
        "price_range": [200, 1200],
        "rationale": "Tech gadgets show linear demand patterns with moderate elasticity. High fixed costs for R&D and tooling. Variable costs reflect component pricing and manufacturing."
    },
    
    "subscription_saas": {
        "industry_name": "SaaS Subscription Services",
        "description": "Software-as-a-Service with recurring revenue model. Price anchoring and value perception are crucial.",
        "demand_type": "Exponential",
        "a": 500,   # Market size for the service
        "b": 1.2,   # Subscription services are price sensitive
        "fixed_cost": 200000,  # Development and infrastructure
        "variable_cost": 5,     # Very low marginal costs per user
        "price_range": [10, 200],
        "rationale": "SaaS exhibits exponential demand decay with price increases. High fixed costs for development but very low marginal costs per additional user."
    },
    
    "pharmaceuticals": {
        "industry_name": "Prescription Pharmaceuticals",
        "description": "Essential medicines with inelastic demand, heavy regulation, and patent protection.",
        "demand_type": "Linear",
        "a": 100,   # Limited by patient population
        "b": 0.3,   # Very low price sensitivity for essential medicines
        "fixed_cost": 2000000,  # Massive R&D and regulatory costs
        "variable_cost": 10,     # Low manufacturing costs
        "price_range": [50, 500],
        "rationale": "Pharmaceuticals have highly inelastic demand due to medical necessity. Enormous fixed costs for R&D and FDA approval, but low manufacturing costs."
    },
    
    "digital_content": {
        "industry_name": "Digital Content & Media",
        "description": "Movies, music, games, and digital media with network effects and zero marginal distribution costs.",
        "demand_type": "Exponential",
        "a": 10000,  # Large potential audience
        "b": 3.0,    # High price sensitivity for entertainment
        "fixed_cost": 1000000,  # High production costs
        "variable_cost": 1,      # Essentially zero distribution costs
        "price_range": [5, 60],
        "rationale": "Digital content has exponential demand curves due to price sensitivity. High upfront production costs but near-zero marginal distribution costs."
    },
    
    "agriculture_commodities": {
        "industry_name": "Agricultural Commodities",
        "description": "Basic food commodities with inelastic demand but high supply volatility and seasonal factors.",
        "demand_type": "Linear",
        "a": 5000,   # Stable food demand
        "b": 0.8,    # Food is relatively inelastic
        "fixed_cost": 50000,   # Farm equipment and land
        "variable_cost": 2,     # Seeds, fertilizer, labor per unit
        "price_range": [1, 10],
        "rationale": "Agricultural commodities have linear, inelastic demand. Moderate fixed costs for farming operations with low variable costs per unit."
    },
    
    "food_delivery": {
        "industry_name": "Food Delivery Services",
        "description": "On-demand food delivery with convenience premium and dynamic pricing based on demand and supply.",
        "demand_type": "Exponential",
        "a": 2000,   # Urban market size
        "b": 2.0,    # Moderate price sensitivity for convenience
        "fixed_cost": 100000,  # App development and marketing
        "variable_cost": 8,     # Delivery costs per order
        "price_range": [3, 25],
        "rationale": "Food delivery shows exponential demand sensitivity to delivery fees. Fixed costs for platform development, variable costs dominated by delivery expenses."
    },
    
    "b2b_enterprise_software": {
        "industry_name": "B2B Enterprise Software",
        "description": "Mission-critical business software with high switching costs and ROI-based pricing.",
        "demand_type": "Linear",
        "a": 200,    # Limited enterprise market
        "b": 0.5,    # Low price sensitivity due to business value
        "fixed_cost": 2000000,  # Major development and sales investment
        "variable_cost": 100,    # Support and implementation per client
        "price_range": [10000, 500000],
        "rationale": "Enterprise software has linear, inelastic demand due to business necessity. Very high fixed costs for development and sales, moderate variable costs for support."
    },
    
    "fashion_retail": {
        "industry_name": "Fashion Retail",
        "description": "Seasonal clothing with trend sensitivity, inventory constraints, and psychological pricing.",
        "demand_type": "Exponential",
        "a": 800,    # Fashion market segment
        "b": 1.8,    # Moderate-high price sensitivity
        "fixed_cost": 200000,  # Inventory, retail space, marketing
        "variable_cost": 25,    # Manufacturing and materials
        "price_range": [20, 300],
        "rationale": "Fashion exhibits exponential demand patterns with seasonal peaks. Fixed costs include inventory risk and retail presence. Variable costs reflect manufacturing."
    },
    
    "ride_sharing": {
        "industry_name": "Ride Sharing Services",
        "description": "Transportation marketplace with surge pricing, network effects, and elastic demand based on convenience vs. cost.",
        "demand_type": "Exponential",
        "a": 3000,   # Urban transportation market
        "b": 2.5,    # High price sensitivity to surge pricing
        "fixed_cost": 500000,  # Platform development and driver acquisition
        "variable_cost": 12,    # Driver payments and fuel costs
        "price_range": [5, 50],
        "rationale": "Ride sharing shows exponential sensitivity to pricing, especially during surge periods. High fixed costs for platform and driver network, variable costs mainly driver compensation."
    },
    
    "cloud_storage": {
        "industry_name": "Cloud Storage Services",
        "description": "Utility-like service with usage-based pricing, high switching costs once adopted, and network effects.",
        "demand_type": "Linear",
        "a": 1500,   # Growing digital storage needs
        "b": 1.0,    # Moderate price sensitivity
        "fixed_cost": 1000000,  # Infrastructure and data centers
        "variable_cost": 0.5,    # Very low marginal storage costs
        "price_range": [5, 100],
        "rationale": "Cloud storage has linear demand with moderate elasticity. Massive fixed infrastructure costs but extremely low marginal costs per GB."
    },
    
    "fitness_memberships": {
        "industry_name": "Fitness & Gym Memberships",
        "description": "Service business with capacity constraints, member retention focus, and psychological commitment factors.",
        "demand_type": "Exponential",
        "a": 600,    # Local market size
        "b": 1.5,    # Price sensitive but value-conscious
        "fixed_cost": 300000,  # Equipment, facility, staff
        "variable_cost": 15,    # Utilities, maintenance per member
        "price_range": [30, 150],
        "rationale": "Fitness memberships show exponential price sensitivity. High fixed costs for facilities and equipment, low variable costs per additional member."
    },
    
    "online_education": {
        "industry_name": "Online Education Platforms",
        "description": "Digital learning with high perceived value, network effects from student communities, and scalable delivery.",
        "demand_type": "Exponential",
        "a": 2500,   # Large addressable market for skills
        "b": 2.2,    # Price sensitive but value-driven
        "fixed_cost": 400000,  # Content creation and platform development
        "variable_cost": 3,     # Minimal delivery costs per student
        "price_range": [50, 500],
        "rationale": "Online education exhibits exponential demand patterns with high price sensitivity. High fixed costs for quality content creation, very low marginal costs per student."
    },
    
    "specialty_coffee": {
        "industry_name": "Specialty Coffee Retail",
        "description": "Premium coffee with brand loyalty, experience-based pricing, and geographic market constraints.",
        "demand_type": "Linear",
        "a": 400,    # Local market with loyal customers
        "b": 1.0,    # Moderate price sensitivity for premium coffee
        "fixed_cost": 150000,  # Equipment, location, staff
        "variable_cost": 2,     # Coffee beans and materials per cup
        "price_range": [3, 12],
        "rationale": "Specialty coffee has linear demand with moderate elasticity due to habit formation and brand loyalty. Fixed costs for premium equipment and locations."
    },
    
    "mobile_gaming": {
        "industry_name": "Mobile Gaming (Freemium)",
        "description": "Free-to-play games with in-app purchases, network effects, and whale-dependent revenue model.",
        "demand_type": "Exponential",
        "a": 50000,  # Large player base potential
        "b": 4.0,    # Very high price sensitivity for virtual goods
        "fixed_cost": 800000,  # Game development and marketing
        "variable_cost": 0.1,   # Server costs per active user
        "price_range": [1, 100],
        "rationale": "Mobile gaming shows extreme exponential sensitivity to IAP pricing. High upfront development costs, very low marginal costs per user."
    }
}

# Utility function to get industry list for dropdown menus
def get_industry_names():
    """Return list of industry names for UI components"""
    return [preset["industry_name"] for preset in INDUSTRY_PRESETS.values()]

# Utility function to get preset by industry name
def get_preset_by_name(industry_name):
    """Get preset dictionary by industry name"""
    for key, preset in INDUSTRY_PRESETS.items():
        if preset["industry_name"] == industry_name:
            return preset
    return None