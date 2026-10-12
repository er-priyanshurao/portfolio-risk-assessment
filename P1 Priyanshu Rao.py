import argparse
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_stock_prices(tickers, start_date, end_date):
    prices = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                if 'Adj Close' in data.columns:
                    prices[ticker] = data['Adj Close']
                elif 'Close' in data.columns:
                    prices[ticker] = data['Close']
        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
            continue
    return prices.dropna()

def calculate_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def calculate_var(returns, weights, alpha=0.05, method='normal'):
    portfolio_returns = returns.dot(weights)

    if method == 'normal':
        mu, sigma = portfolio_returns.mean(), portfolio_returns.std()
        z_alpha = stats.norm.ppf(alpha)
        return -(mu + sigma * z_alpha)

    elif method == 't':
        df, mu_t, sigma_t = stats.t.fit(portfolio_returns)
        t_alpha = stats.t.ppf(alpha, df)
        return -(mu_t + sigma_t * t_alpha)

    elif method == 'historical':
        return -np.percentile(portfolio_returns, alpha * 100)

    else:
        raise ValueError("Method must be 'normal', 't', or 'historical'")

def calculate_diversification_benefit(returns, weights):
    individual_vol = returns.std() * weights
    portfolio_vol = returns.dot(weights).std()
    return individual_vol.sum() - portfolio_vol

def maximum_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def test_normality(returns):
    return stats.jarque_bera(returns)

def test_mean_return(returns):
    return stats.ttest_1samp(returns, 0)

def plot_correlation_matrix(returns):
    corr = returns.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Stock Return Correlations')
    plt.show()

def plot_rolling_var(returns, weights, window=60):
    portfolio_returns = returns.dot(weights)
    rolling_var = portfolio_returns.rolling(window).apply(
        lambda x: -np.percentile(x, 5))

    plt.figure(figsize=(10, 6))
    plt.plot(rolling_var)
    plt.title(f'Rolling {window}-Day 95% VaR')
    plt.xlabel('Date')
    plt.ylabel('VaR')
    plt.grid(True)
    plt.show()

def plot_cumulative_returns(returns, weights):
    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns)
    plt.title('Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Growth of $1')
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Portfolio Risk Analysis Tool')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOG'])
    parser.add_argument('--weights', nargs='+', type=float, default=[0.3, 0.4, 0.3])
    parser.add_argument('--start', default='2020-01-01')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()

    if not np.isclose(sum(args.weights), 1.0):
        raise ValueError("Weights must sum to 1.0")

    print("\nFetching stock data...")
    prices = fetch_stock_prices(args.tickers, args.start, args.end)
    if prices.empty:
        raise ValueError("No data fetched - check your ticker symbols and date range")

    returns = calculate_log_returns(prices)
    portfolio_returns = returns.dot(args.weights)

    print("\nCalculating Value at Risk (VaR):")
    var_normal = calculate_var(returns, args.weights, method='normal')
    var_t = calculate_var(returns, args.weights, method='t')
    var_hist = calculate_var(returns, args.weights, method='historical')

    print(f"Normal VaR (95%): {var_normal:.4f}")
    print(f"t-Distribution VaR (95%): {var_t:.4f}")
    print(f"Historical VaR (95%): {var_hist:.4f}")

    div_benefit = calculate_diversification_benefit(returns, args.weights)
    print(f"\nDiversification Benefit: {div_benefit:.4f}")

    print("\nStatistical Tests:")
    jb_stat, jb_p = test_normality(portfolio_returns)
    t_stat, t_p = test_mean_return(portfolio_returns)

    print(f"Jarque-Bera Test (Normality): Stat={jb_stat:.2f}, p-value={jb_p:.4f}")
    print(f"t-Test (Mean Return): Stat={t_stat:.2f}, p-value={t_p:.4f}")

    cumulative_returns = (1 + portfolio_returns).cumprod()
    mdd = maximum_drawdown(cumulative_returns)
    print(f"\nMaximum Drawdown: {mdd:.2%}")

    print("\nGenerating visualizations...")
    plot_correlation_matrix(returns)
    plot_rolling_var(returns, args.weights)
    plot_cumulative_returns(returns, args.weights)

if __name__ == '__main__':
    main()