import pandas as pd
import numpy as np
import torch
import json
import logging

from pytorch_forecasting.models import TemporalFusionTransformer

# Suppress unnecessary warnings from the lightning module
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


def predict(new_data_df, encoder_length, num_steps_to_predict, params_path: str = 'Model/All Features/ Weights & Parameters/best_hyperparameters.json', weights_path: str = 'Model/All Features/ Weights & Parameters/weights_model.pth', dataset_path: str = 'Model/All Features/ Weights & Parameters/training_dataset.pth'):
    """
    Loads a pre-trained Temporal Fusion Transformer model and uses it to generate
    return predictions for a given set of time series data.

    Args:
        new_data_df (pd.DataFrame): DataFrame containing the input data for prediction.
                                    Must include the encoder history for each series.
        encoder_length (int): The number of past time steps the model uses as input.
        num_steps_to_predict (int): The number of future time steps to predict.
        params_path (str): Path to the JSON file with the model's hyperparameters.
        weights_path (str): Path to the .pth file with the saved model weights.
        dataset_path (str): Path to the .pth file with the saved TimeSeriesDataSet
                            object, which contains necessary metadata.

    Returns:
        dict: A dictionary where keys are the group IDs (e.g., CUSIPs) and values are
              dictionaries containing the predicted returns, actual returns for
              comparison, and corresponding time indices.
    """
    # --- 1. Load Model and Configuration ---
    # Load the dataset structure which contains metadata like normalizers
    training = torch.load(dataset_path, weights_only=False) 
    
    # Load the best hyperparameters found during optimization
    with open(params_path, 'r') as f:
            loaded_best_params = json.load(f)
    
    # Recreate the model architecture with the same hyperparameters
    attention_head_size = loaded_best_params.get('attention_head_size')
    hidden_size_multiple = loaded_best_params.get('hidden_size_multiple')
    hidden_size = attention_head_size * hidden_size_multiple

    loaded_model = TemporalFusionTransformer.from_dataset(
        training, 
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=loaded_best_params.get('dropout'),
        hidden_continuous_size=loaded_best_params.get('hidden_continuous_size'),
        learning_rate=loaded_best_params.get('learning_rate'),
        output_size=1
    )

    # Load the trained weights into the model architecture
    loaded_model.load_state_dict(torch.load(weights_path))
    loaded_model.eval() # Set model to evaluation mode (disables dropout, etc.)
    
    # --- 2. Iterate and Predict for Each Stock ---
    all_predictions_output = {}
    group_id_column = training.group_ids[0] # Get the name of the stock identifier column (e.g., 'CUSIP')

    # Get a list of all unique stocks (CUSIPs) in the prediction dataset
    unique_groups_in_new_data = new_data_df[group_id_column].unique()

    # Loop through each stock to make individual predictions
    for group_id_val in unique_groups_in_new_data:
        # Get the historical data for the current stock
        group_specific_new_data = new_data_df[new_data_df[group_id_column] == group_id_val].copy()
        group_specific_new_data.sort_values(by="time_idx", inplace=True)

        # --- 3. Prepare Input Data for the Model ---
        # The model needs a specific length of data to make a prediction:
        # It takes 'encoder_length' of past data to predict 'prediction_length' of future data.
        model_max_configured_pred_len = training.max_prediction_length
        actual_steps_to_extract = min(num_steps_to_predict, model_max_configured_pred_len)
        required_length_for_predict_call = encoder_length + actual_steps_to_extract

        # Slice the dataframe to get the exact amount of data needed by the model
        data_for_model_predict = group_specific_new_data.iloc[:required_length_for_predict_call].copy()
            
        # --- 4. Make Prediction ---
        with torch.no_grad(): # Disable gradient calculations for inference
            prediction_output_object = loaded_model.predict(
                data_for_model_predict,
                mode="raw" # Use 'raw' mode to get the direct tensor output
            )
        
        # --- 5. Extract and Format Results ---
        # Extract the prediction tensor from the output object
        actual_raw_prediction_tensor = prediction_output_object.prediction
        # Get the predicted values for the future steps and convert to a numpy array
        predicted_target_values = actual_raw_prediction_tensor[0, :actual_steps_to_extract, 0].cpu().numpy()
        
        # Find the last time step used in the encoder to align actuals and predictions
        last_encoder_time_idx = group_specific_new_data['time_idx'].iloc[encoder_length - 1]
        
        predicted_time_indices = []
        actual_target_values_for_comparison = []

        # Loop through the prediction window to get the actual returns for comparison
        for i in range(actual_steps_to_extract):
            current_pred_time_idx = last_encoder_time_idx + 1 + i
            predicted_time_indices.append(current_pred_time_idx)
            
            # Find the row with the actual return for the time step we just predicted
            actual_row = group_specific_new_data[group_specific_new_data['time_idx'] == current_pred_time_idx]
            if not actual_row.empty:
                actual_target_values_for_comparison.append(actual_row['Stock_return'].iloc[0])
            else:
                actual_target_values_for_comparison.append(np.nan) # Append NaN if no actual value exists

        # Store the results for the current stock in a dictionary
        all_predictions_output[group_id_val] = {
            "predicted_returns": predicted_target_values,
            "actual_returns": np.array(actual_target_values_for_comparison),
            "time_indices": predicted_time_indices,
            "encoder_last_time_idx": last_encoder_time_idx
        }
    return all_predictions_output

def run_prediction():
    """
    Acts as the main execution script. It defines the prediction parameters,
    prepares the holdout dataset, calls the prediction function, and formats
    the final output into a single CSV file for analysis.
    """
    # Load the full dataset
    data = pd.read_csv('final_data_reduced_non-normalized.csv')
    date_map = data[['time_idx', 'Date']].drop_duplicates().set_index('time_idx')['Date']    
    
    # Define model parameters
    encoder_history_length = 12 # Model looks at 12 months of history
    num_predict_steps = 3       # Model predicts 3 months into the future
    required_data_length = encoder_history_length + num_predict_steps
    
    # For each stock (CUSIP), get the last 15 months of data for prediction
    prediction_data = data.groupby('CUSIP').tail(required_data_length).copy()
    prediction_data.sort_values(by=['CUSIP', "time_idx"], inplace=True)
    
    # Call the main predict function
    predictions = predict(
        new_data_df=prediction_data,
        encoder_length=encoder_history_length,
        num_steps_to_predict=num_predict_steps
    )

    # Process the results and save them to a single CSV file
    if predictions:
        all_results_df = []
        for company, result in predictions.items():
            df = pd.DataFrame({
                "time_idx": result["time_indices"],
                "actual_return": result["actual_returns"],
                "predicted_return": result["predicted_returns"]
            })
            df['CUSIP'] = company
            all_results_df.append(df)
        
        final_df = pd.concat(all_results_df, ignore_index=True)
        final_df['Date'] = final_df['time_idx'].map(date_map)
        
        output_filename = 'predictions.csv'
        final_df.to_csv(output_filename, index=False)
        print(f"\n--- Prediction results saved to {output_filename} ---")
        print(final_df.head())

if __name__ == "__main__":
    run_prediction()
