from pytorch_forecasting import TimeSeriesDataSet 
from pytorch_forecasting.data import GroupNormalizer 

def data_loader(data):
    """
    Transforms a pandas DataFrame into PyTorch DataLoaders suitable for training
    the Temporal Fusion Transformer model.

    This function defines the model's input/output structure, splits the data
    into training and validation sets, and wraps them in TimeSeriesDataSet and
    DataLoader objects.

    Args:
        data (pd.DataFrame): The input DataFrame containing time series data for
                             all stocks. Must include columns for time index,
                             target variable, group ID, and all features.

    Returns:
        tuple: A tuple containing:
            - train_dataloader (DataLoader): DataLoader for the training set.
            - val_dataloader (DataLoader): DataLoader for the validation set.
            - training (TimeSeriesDataSet): The training TimeSeriesDataSet object.
            - validation (TimeSeriesDataSet): The validation TimeSeriesDataSet object.
    """
    # --- 1. Define Model and Data Structure ---
    # Define the length of the prediction window (3 months)
    max_prediction_length = 3
    # Define the length of the historical window the model sees (12 months)
    max_encoder_length = 12
    # Determine the last time step for the training data
    training_cutoff = data.time_idx.max() - max_prediction_length

    # Define the names of key columns
    target_col = 'Stock_return' # The variable we want to predict
    group_col = 'CUSIP'          # The column that identifies each individual stock

    # Define the list of all features to be used as model inputs.
    # These are variables whose future values are not known at prediction time.
    time_varying_unknown_reals = ['Stock_return',
                                 'SP500_return',
                                 'Firm age',
                                 'Liquidity of market assets',
                                 'Amihud Measure',
                                 'Capital turnover',
                                 'Change in common equity',
                                 'Book-to-market equity',
                                 'Dimson beta',
                                 'Frazzini-Pedersen market beta',
                                 'CAPEX growth (1 year)',
                                 'Cash-to-assets',
                                 'Net stock issues',
                                 'Change in current operating assets',
                                 'Cash-based operating profits-to-book assets',
                                 'Cash-based operating profits-to-lagged book assets',
                                 'Dividend yield',
                                 'Dollar trading volume',
                                 'Change sales minus change Inventory',
                                 'Change sales minus change receivables',
                                 'Earnings variability',
                                 'Ebitda-to-market enterprise value',
                                 'Equity net payout',
                                 'Payout yield',
                                 'Free cash flow-to-price',
                                 'Gross profits-to-lagged assets',
                                 'Inventory growth',
                                 'Intrinsic value-to-market',
                                 'Idiosyncratic volatility from the CAPM (21 days)',
                                 'Idiosyncratic volatility from the CAPM (252 days)',
                                 'Idiosyncratic volatility from the Fama-French 3-factor model',
                                 'Idiosyncratic volatility from the q-factor model',
                                 'Kaplan-Zingales index',
                                 'Change in long-term net operating assets',
                                 'Mispricing factor: Management',
                                 'Mispricing factor: Performance',
                                 'Change in net financial assets',
                                 'Return on equity',
                                 'Earnings volatility',
                                 'Earnings-to-price',
                                 'Quarterly return on assets',
                                 'Quarterly return on equity',
                                 'Change in net noncurrent operating assets',
                                 'Ohlson O-score',
                                 'Operating accruals',
                                 'Percent operating accruals',
                                 'Operating cash flow-to-market',
                                 'Operating profits-to-lagged book assets',
                                 'Operating profits-to-book equity',
                                 'Operating leverage',
                                 'Price per share',
                                 'Current price to high price over last year',
                                 'Quality minus Junk: Profitability',
                                 'Quality minus Junk: Safety',
                                 'R&D-to-sales',
                                 'Price momentum t-12 to t-1',
                                 'Short-term reversal',
                                 'Long-term reversal',
                                 'Price momentum t-9 to t-1',
                                 'Maximum daily return',
                                 'Sales Growth (1 year)',
                                 'Year 1-lagged return, nonannual',
                                 'Years 2-5 lagged returns, nonannual',
                                 'Total accruals',
                                 'Share turnover',
                                 'Number of zero trades with turnover as tiebreaker (6 months)',
                                 'Number of zero trades with turnover as tiebreaker (1 month)',
                                 'Number of zero trades with turnover as tiebreaker (12 months)',
                                 'TMAX_SODAKOTA',
                                 'TMIN_SODAKOTA',
                                 'TMAX_INDIANA',
                                 'TMIN_INDIANA',
                                 'AWND_ILLINOIS',
                                 'TMAX_KANSAS',
                                 'TMIN_KANSAS',
                                 'TMAX_NEBRASKA',
                                 'TMIN_NEBRASKA',
                                 'SNWD_IOWA',
                                 'TMAX_IOWA',
                                 'TMIN_IOWA',
                                 'TMAX_MINNESOTA',
                                 'TMIN_MINNESOTA']
    
    # --- 2. Create TimeSeriesDataSet for Training ---
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        allow_missing_timesteps=True,
        time_idx="time_idx",                      # Column that identifies the time step
        target=target_col,                        # Column with the value to predict
        group_ids=[group_col],                    # Column that identifies each time series
        min_encoder_length=max_encoder_length,    # Minimum history required for a sample
        max_encoder_length=max_encoder_length,    # Maximum history used for a sample
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[group_col],          # Features that are constant for each stock (here, just the ID)
        time_varying_known_reals=["time_idx"],    # Features whose future values are known (e.g., month number)
        time_varying_unknown_reals=time_varying_unknown_reals, # The main input features
        target_normalizer=GroupNormalizer(
            groups=[group_col] # Normalize the target variable separately for each stock
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # --- 3. Create Validation Set and DataLoaders ---
    # Create the validation set from the full dataset, using the same settings as the training set
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    # Define the batch size for training
    batch_size = 256

    # Create the PyTorch DataLoader objects that will feed data to the model in batches
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=12)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 5, num_workers=12)
    
    return train_dataloader, val_dataloader, training, validation
