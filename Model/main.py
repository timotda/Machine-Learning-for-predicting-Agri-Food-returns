import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore") 
import json

# Import the necessary functions from other project files
from create_dataloader import data_loader 
from TFT import train_tft_model 
from hyperpar_opt import run_optuna_hpo

if __name__ == "__main__":
    """
    Main execution block to run the end-to-end model training pipeline.

    This script orchestrates the entire process:
    1. Loads the dataset.
    2. Splits data into training and a holdout set for later prediction.
    3. Runs hyperparameter optimization using Optuna to find the best model settings.
    4. Trains the final model using the best hyperparameters found.
    5. Saves the trained model weights and configuration for future use.
    """
    
    # --- 1. Data Loading and Preparation ---
    # Load the full, pre-processed dataset
    data = pd.read_csv('data/Data Files/final_data_reduced_non-normalized.csv')
    
    # Define the encoder (history) and prediction lengths for the model
    encoder_length = 12
    prediction_length = 3
    
    # Create the holdout set: the last 15 months of data for each stock (CUSIP)
    # This data is NOT used in training or validation and is reserved for final testing.
    holdout_cutoff = encoder_length + prediction_length 
    holdout_set = data.groupby("CUSIP").tail(holdout_cutoff)
    
    # Create the training dataset by removing the holdout set
    training_data = data.drop(holdout_set.index).copy()
    
    # Create PyTorch DataLoaders for training and validation
    train_dl, val_dl, training, validation = data_loader(training_data)
    
    # --- 2. Hyperparameter Optimization (HPO) ---
    # Save the training TimeSeriesDataSet object. This is crucial because it contains
    # metadata like normalizers that are needed for making predictions later.
    # NOTE: Comment/uncomment these lines to switch between 'All Features' and 'Restricted Features' runs.
    # dataset_path = 'Model/All Features/Weights & Parameters/training_dataset.pth'
    dataset_path = 'Model/Restricted Features/Weights & Parameters/training_dataset.pth'
    torch.save(training, dataset_path)
    
    # Run the Optuna HPO study to find the best model parameters
    print("--- Starting Hyperparameter Optimization ---")
    best_trial = run_optuna_hpo(
        training=training,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        n_trials=40, # Number of different hyperparameter combinations to test
    )
    print("\n--- Hyperparameter Optimization Finished ---")
    print(f"Best trial validation loss: {best_trial.value}")
    print(f"Best parameters: {best_trial.params}")
    
    # Save the best found parameters to a JSON file
    # NOTE: Comment/uncomment these lines to switch between 'All Features' and 'Restricted Features' runs.
    # best_params_path = 'Model/All Features/Weights & Parameters/best_hyperparameters.json'
    best_params_path = 'Model/Restricted Features/Weights & Parameters/best_hyperparameters.json'
    with open(best_params_path, 'w') as f:
        json.dump(best_trial.params, f, indent=4)
        
    # --- 3. Final Model Training ---
    # Train the final model on the full training dataset for 200 epochs
    # using the best hyperparameters discovered by Optuna.
    print("\n--- Starting Final Model Training ---")
    trained_model = train_tft_model(
        train_dl, 
        val_dl, 
        training, 
        best_trial.params
    )   
    
    # Save the state dictionary (the learned weights) of the final trained model.
    # This file is used later by predict_returns.py to make predictions.
    # NOTE: Comment/uncomment these lines to switch between 'All Features' and 'Restricted Features' runs.
    # torch.save(trained_model.state_dict(), 'Model/All Features/Weights & Parameters/weights_model.pth') 
    torch.save(trained_model.state_dict(), 'Model/Restricted Features/Weights & Parameters/weights_model.pth') 
    
    print("\n--- Pipeline Complete. Model and parameters saved. ---")
