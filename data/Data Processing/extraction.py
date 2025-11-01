import pandas as pd
from pandas import DataFrame, DatetimeIndex
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt


def load_and_process_file(filepath:str,
                          state_name: str,
                          measurements: List[str],
                          date_range: DatetimeIndex) -> DataFrame:
    
    """
    Load a CSV weather data file, filter specific columns, reindex by date, and rename columns with state suffix.

    Parameters:
        filepath (str): Path to the CSV file.
        state_name (str): State name to append as suffix to column names.
        measurements (List[str]): List of column names to retain.
        date_range (DatetimeIndex): Full date index to reindex the DataFrame.
    Returns:
        DataFrame: Cleaned and reindexed DataFrame with renamed columns.
    """
    df = pd.read_csv(filepath)
    
    filtered_cols = [col for col in df.columns if col.upper() in measurements]
    df = df[filtered_cols]
    
    df.columns = [col.upper() for col in df.columns]
    
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df = df.reindex(date_range)
    
    # Add state suffix to columns
    df = df.add_suffix(f"_{state_name.upper()}")
    
    return df


def nan_summary(df:DataFrame) -> None:
    """
    Display a summary bar chart of the proportion of NaN values in each column.

    Parameters:
        df (DataFrame): Input DataFrame.
    Returns:
        None (only displays a plot).
    """    
    nan_count = df.isna().sum()
    nan_proportion = df.isna().mean()
    summary = pd.DataFrame({
        'NaN Count': nan_count,
        'NaN Proportion': nan_proportion
    })
    # Plotting
    plt.figure(figsize=(15, 3))
    plt.bar(summary.index, summary['NaN Proportion'], color='darkred')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.xlabel('Measure')
    plt.ylabel('NaN Proportion')
    plt.title('Proportion of NaN values')
    plt.grid(True)
    plt.show()
    return


def cut_data(df: DataFrame,
             start_date: datetime,
             end_date: datetime) -> DataFrame:
    """
    Slice the DataFrame between two datetime bounds based on its datetime index.

    Parameters:
        df (DataFrame): Input DataFrame with a datetime index named 'DATE'.
        start_date (datetime): Start date of the time window.
        end_date (datetime): End date of the time window.
    Returns:
        DataFrame: Sliced DataFrame within the specified date range.
    """
    if not (pd.api.types.is_datetime64_any_dtype(df.index) and df.index.name == "DATE"):
        raise ValueError("DataFrame index must be datetime-formatted and named 'DATE'")
    
    df_short = df[(df.index >= start_date) & (df.index <= end_date)]
    return df_short