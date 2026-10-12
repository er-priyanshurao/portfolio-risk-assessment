# portfolio-risk-assessment

# Portfolio Risk Assessment & Value at Risk (VaR) Calculator

A Python-based financial risk analysis tool designed to evaluate stock portfolio risk using historical market data and statistical techniques.

The project calculates Value at Risk (VaR) using three different methods and provides additional insights into portfolio diversification, maximum drawdown, return distributions, and stock correlations.

## Features

- **Stock Market Data Retrieval:** Fetch historical stock prices using Yahoo Finance.
- **Value at Risk (VaR):** Calculate 95% VaR using:
  - Normal Distribution Method
  - Student's t-Distribution Method
  - Historical Simulation Method
- **Portfolio Return Analysis:** Calculate portfolio returns using user-defined asset weights.
- **Diversification Benefit:** Estimate the reduction in portfolio volatility from diversification.
- **Maximum Drawdown:** Measure the largest peak-to-trough decline in portfolio value.
- **Statistical Testing:**
  - Jarque-Bera test for return normality.
  - One-sample t-test for mean returns.
- **Correlation Analysis:** Visualize correlations between stock returns.
- **Rolling VaR:** Analyze changes in historical portfolio risk over time.
- **Cumulative Returns:** Visualize portfolio growth over the selected period.
- **Command-Line Interface:** Customize tickers, portfolio weights, and date ranges.

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- yfinance

## Project Structure

portfolio-risk-assessment/
│
├── portfolio_risk.py
├── requirements.txt
└── README.md

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/er-priyanshurao/portfolio-risk-assessment.git
cd portfolio-risk-assessment
