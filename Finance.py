import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Fetch Stock Data
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
stock_data = yf.download(tickers, start="2018-01-01", end="2024-08-01")['Adj Close']

print("Stock Data (First 5 rows):")
print(stock_data.head())

# Step 2: Calculate Daily Returns
daily_returns = stock_data.pct_change().dropna()
print("\nDaily Returns (First 5 rows):")
print(daily_returns.head())

# Step 3: Portfolio Statistics (Initial Equal-Weight Portfolio)
weights = [1/len(tickers)] * len(tickers)

expected_return = np.sum(daily_returns.mean() * weights) * 252  # Annualized return
portfolio_volatility = np.sqrt(np.dot(weights, np.dot(daily_returns.cov() * 252, weights)))  # Annualized volatility
sharpe_ratio = expected_return / portfolio_volatility

print(f"\nInitial Portfolio - Expected Return: {expected_return}")
print(f"Initial Portfolio - Volatility: {portfolio_volatility}")
print(f"Initial Portfolio - Sharpe Ratio: {sharpe_ratio}")

# Step 4: Efficient Frontier (Monte Carlo Simulation)
num_portfolios = 10000
results = np.zeros((3, num_portfolios))  # Store portfolio return, volatility, and Sharpe ratio

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    
    port_return = np.sum(daily_returns.mean() * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
    sharpe = port_return / port_volatility
    
    # Save the results
    results[0, i] = port_volatility
    results[1, i] = port_return
    results[2, i] = sharpe

# Step 5: Plot Efficient Frontier
plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier')
plt.show()

# Step 6: Optimal Portfolio (Max Sharpe Ratio)
max_sharpe_idx = np.argmax(results[2, :])  # Portfolio with maximum Sharpe ratio
optimal_volatility = results[0, max_sharpe_idx]
optimal_return = results[1, max_sharpe_idx]

print(f"\nOptimal Portfolio - Expected Return: {optimal_return}")
print(f"Optimal Portfolio - Volatility: {optimal_volatility}")

# Step 7: Allocation Recommendation
optimal_weights = np.random.random(len(tickers))
optimal_weights /= np.sum(optimal_weights)  # Normalize weights to sum to 1

allocation = pd.DataFrame({'Stock': tickers, 'Allocation': optimal_weights})
print("\nOptimal Portfolio Allocation:")
print(allocation)
