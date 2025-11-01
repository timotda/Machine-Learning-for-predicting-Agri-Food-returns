import pandas as pd
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt 

def replacing_outliers(data):
    """
    Replaces outlier values in a pandas Series with the median.

    Any value greater than 0.15 is considered an outlier and is replaced
    by the median of the series.

    Args:
        data (pd.Series): A pandas Series of numerical data.

    Returns:
        pd.Series: The data with outliers replaced.
    """
    median = data.median()
    data = data.where(data < 0.15, median)
    return data

def get_predictions_from_file(filepath):
    """
    Loads prediction data from a CSV file and pivots it into a format
    suitable for backtesting.

    Args:
        filepath (str): The path to the predictions CSV file.

    Returns:
        tuple: A tuple containing two DataFrames:
            - predicted_returns_df (pd.DataFrame): Predicted returns with time_idx
                                                    as index and CUSIPs as columns.
            - actual_returns_df (pd.DataFrame): Actual returns in the same format.
    """
    data = pd.read_csv(filepath)
    # Reshape the data into the required format (rows=time, columns=assets)
    predicted_returns_df = data.pivot(
        index='time_idx', columns='CUSIP', values='predicted_return'
    )
    actual_returns_df = data.pivot(
        index='time_idx', columns='CUSIP', values='actual_return'
    )
    
    return predicted_returns_df, actual_returns_df

def calculate_historical_covariance(data, assets):
    """
    Calculates the historical covariance matrix for a given set of assets.

    Args:
        data (pd.DataFrame): DataFrame containing the full historical stock returns.
        assets (list): A list of CUSIPs to include in the covariance matrix.

    Returns:
        pd.DataFrame: The covariance matrix of asset returns.
    """
    historical_data = data[data['CUSIP'].isin(assets)]
    returns_pivot = historical_data.pivot(index='time_idx', columns='CUSIP', values='Stock_return')
    return returns_pivot.cov()

def strategy_long_short_equal_weight(predicted_returns, long_threshold=0.001, short_threshold=-0.001):
    """
    Constructs a dollar-neutral, long-short portfolio with equal weights.

    Args:
        predicted_returns (pd.Series): A Series of predicted returns for each asset.
        long_threshold (float): The minimum predicted return to go long.
        short_threshold (float): The maximum predicted return to go short.

    Returns:
        pd.Series: A Series of portfolio weights for each asset.
    """
    # Identify assets to long and short based on thresholds
    assets_to_long = predicted_returns[predicted_returns > long_threshold].index
    assets_to_short = predicted_returns[predicted_returns < short_threshold].index
    
    weights = pd.Series(0.0, index=predicted_returns.index)
    
    # Assign equal weights to long positions (totaling 100%)
    if len(assets_to_long) > 0:
        long_weight = 1.0 / len(assets_to_long)
        weights[assets_to_long] = long_weight
        
    # Assign equal weights to short positions (totaling -100%)
    if len(assets_to_short) > 0:
        short_weight = -1.0 / len(assets_to_short)
        weights[assets_to_short] = short_weight
        
    return weights

def compute_sharpe_ratio(returns, risk_free):
    """
    Calculates the Sharpe ratio for a set of strategy returns.

    Args:
        returns (pd.DataFrame): A DataFrame of periodic returns for each strategy.
        risk_free (pd.DataFrame): DataFrame containing historical risk-free rates.

    Returns:
        pd.Series: A Series of Sharpe ratios for each strategy.
    """
    means = returns.mean(axis=0)
    vol = returns.std(axis=0)
    sharpe = (means-risk_free['GS1M'].mean())/vol
    return sharpe

def calculate_r_squared(actual_returns, predicted_returns):
    """
    Calculates the R-squared value for each asset.

    Args:
        actual_returns (pd.DataFrame): DataFrame of actual returns.
        predicted_returns (pd.DataFrame): DataFrame of predicted returns.

    Returns:
        pd.Series: A Series of R-squared scores for each asset (CUSIP).
    """
    # Combine actual and predicted returns into a single DataFrame and drop NaNs
    aligned_data = pd.DataFrame({
        'actual': actual_returns.stack(),
        'predicted': predicted_returns.stack()
    }).dropna()
    r_squared_scores = {}
    
    # Calculate R-squared for each asset individually
    for cusip in actual_returns.columns.intersection(predicted_returns.columns):
        asset_data = aligned_data.loc[pd.IndexSlice[:, cusip], :]
        if asset_data.empty:
            r_squared_scores[cusip] = np.nan
            continue

        y_true = asset_data['actual'].values
        y_pred = asset_data['predicted'].values
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
      
        r_squared_scores[cusip] = 1 - (ss_res / ss_tot)
            
    return pd.Series(r_squared_scores)

def strategy_mean_variance_optimization(predicted_returns, cov_matrix, risk_free_rate=0.0):
    """
    Calculates the optimal portfolio weights using Mean-Variance Optimization
    to find the tangency portfolio (which maximizes the Sharpe ratio).

    Args:
        predicted_returns (pd.Series): A Series of predicted returns for each asset.
        cov_matrix (pd.DataFrame): The historical covariance matrix of asset returns.
        risk_free_rate (float): The risk-free rate for the period.

    Returns:
        pd.Series: A Series of optimal portfolio weights for each asset.
    """
    cleaned_returns = predicted_returns.dropna()
    assets = cleaned_returns.index
    mu = cleaned_returns.values.reshape(-1, 1)
    Sigma = cov_matrix.loc[assets, assets].values
    
    # Use the analytical solution for the tangency portfolio
    Sigma_inv = np.linalg.inv(Sigma)
    excess_returns = mu - risk_free_rate
    numerator = np.dot(Sigma_inv, excess_returns)
    denominator = np.sum(numerator)
    weights = numerator / denominator
    return pd.Series(weights.flatten(), index=assets)

def run_backtest(restricted_features=True):
    """
    Main function to run the entire backtesting simulation.

    This function loads data, sets up the backtest parameters, iterates through
    each time step to simulate the strategies, and calculates and prints the
    final performance metrics and plots.
    """
    # --- 1. Load Data and Setup ---
    if restricted_features:
        predictions_paths = 'Model/Restricted Features/Results/predictions.csv'
    else: 
        predictions_paths = 'Model/All Features/Results/predictions.csv'
    predicted_returns, actual_returns = get_predictions_from_file(predictions_paths)
    
    # Load other necessary data
    predictions_with_date = pd.read_csv(predictions_paths)
    risk_free = pd.read_csv('data/Data Files/risk-free.csv')
    risk_free['GS1M'] = risk_free['GS1M'] / (100*12) # Convert to monthly rate
    risk_free['observation_date'] = pd.to_datetime(risk_free['observation_date'])
    date_map = predictions_with_date[['time_idx', 'Date']].drop_duplicates().set_index('time_idx')['Date']
    all_historical_data = pd.read_csv('data/Data Files/final_data_reduced_non-normalized.csv')
    assets = predicted_returns.columns
    
    # --- 2. Pre-computations ---
    cov_matrix = calculate_historical_covariance(all_historical_data, assets)
    r_squared_scores = calculate_r_squared(actual_returns, predicted_returns)
    
    # --- 3. Run Backtest Simulation Loop ---
    portfolio_results = []
    # Iterate through each prediction step (each month in the holdout period)
    for i in range(len(predicted_returns)):
        time_idx_label = predicted_returns.index[i]
        step_results = {"Time Index": time_idx_label}
        
        # Get predictions and actuals for the current time step
        preds_step_i = predicted_returns.iloc[i]
        actuals_step_i = actual_returns.iloc[i]
        
        # Get the current risk-free rate
        current_date = date_map.loc[time_idx_label]
        current_risk_free_rate = risk_free[risk_free['observation_date'] <= current_date]['GS1M'].iloc[-1]

        # Calculate portfolio weights for each strategy
        weights_ls = strategy_long_short_equal_weight(preds_step_i)
        weights_mvo = strategy_mean_variance_optimization(preds_step_i, cov_matrix, current_risk_free_rate)

        # Calculate the realized return for each strategy for this period
        step_results['LS_Return'] = np.sum(weights_ls * actuals_step_i)
        step_results['MVO_Return'] = np.sum(weights_mvo * actuals_step_i) + risk_free.iloc[i]['GS1M']
        step_results['Benchmark_Return'] = actuals_step_i.mean()
        
        portfolio_results.append(step_results)

    results_df = pd.DataFrame(portfolio_results).set_index("Time Index")

    # --- 4. Calculate Final Performance Metrics ---
    cumulative_perf = results_df.copy()
    # Use geometric compounding for MVO and Benchmark strategies
    for col in ['MVO_Return','Benchmark_Return']:
        cumulative_perf[col] = (1 + cumulative_perf[col]).cumprod() - 1
    # Use additive sum for the dollar-neutral long-short strategy
    cumulative_perf['LS_Return'] = cumulative_perf['LS_Return'].cumsum()

    print("\n--- Cumulative Portfolio Performance ---")
    print(cumulative_perf.iloc[-1])

    # Map time index to actual dates for plotting
    cumulative_perf.index = cumulative_perf.index.map(date_map)
    cumulative_perf.index = pd.to_datetime(cumulative_perf.index) 

    sharpe_ratios = compute_sharpe_ratio(results_df, risk_free)
    print("\n--- Sharpe Ratios ---")
    print(sharpe_ratios)

    # --- 5. Plot Results ---
    plt.figure(figsize=(15, 8))
    for strategy_name in cumulative_perf.columns:
        plt.plot(cumulative_perf.index, cumulative_perf[strategy_name], label=strategy_name)
    plt.title('Cumulative Strategy Performance', fontsize=20)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.xlabel('Date', fontsize=14) 
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_backtest(restricted_features=False)
