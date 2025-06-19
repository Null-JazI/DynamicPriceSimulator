import matplotlib.pyplot as plt

def plot_revenue_profit(prices, revenue, profit, optimal_price=None, user_price=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(prices, revenue, label='Revenue ($)', color='blue', linewidth=2)
    ax.plot(prices, profit, label='Profit ($)', color='green', linewidth=2)

    if optimal_price:
        ax.axvline(optimal_price, color='purple', linestyle='--', label=f'Optimal Price (${optimal_price:.2f})')

    if user_price:
        ax.axvline(user_price, color='red', linestyle=':', label=f'Your Price (${user_price:.2f})')

    ax.set_title("Revenue and Profit vs. Price", fontsize=16)
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Dollars ($)")
    ax.legend()
    ax.grid(True)

    return fig
