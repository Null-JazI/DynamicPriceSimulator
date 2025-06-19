def evaluate_price_optimality(user_price, optimal_price, tolerance_percent=5):
    """
    Returns True if the user's price is within the acceptable range of the optimal price.
    """
    if optimal_price is None:
        return False  # Can't evaluate if optimal price wasn't found

    lower_bound = optimal_price * (1 - tolerance_percent / 100)
    upper_bound = optimal_price * (1 + tolerance_percent / 100)

    return lower_bound <= user_price <= upper_bound


def generate_recommendation(is_optimal, user_price, optimal_price):
    """
    Returns a suggestion string based on pricing comparison.
    """
    if optimal_price is None:
        return "⚠️ Optimal price could not be determined. Please check your inputs or try a different model."

    if is_optimal:
        return "✅ Your current price is close to the optimal value. Good job!"

    # Below optimal
    if user_price < optimal_price:
        percent_off = round(((optimal_price - user_price) / optimal_price) * 100, 2)
        return f"🔼 Consider raising your price by approximately {percent_off}% to increase profitability."

    # Above optimal
    elif user_price > optimal_price:
        percent_off = round(((user_price - optimal_price) / optimal_price) * 100, 2)
        return f"🔽 Your price may be too high. Reducing it by about {percent_off}% could boost revenue and demand."

    return "🤔 Unable to generate a recommendation. Please check values."
