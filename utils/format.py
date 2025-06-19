def format_currency(value, decimals=2):
    """
    Formats a number as currency, e.g., 1234.56 → $1,234.56
    """
    if value is None:
        return "N/A"
    return f"${value:,.{decimals}f}"


def format_percent(value, decimals=2):
    """
    Formats a number as a percentage, e.g., 0.245 → 24.50%
    """
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"
