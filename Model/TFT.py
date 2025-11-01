import torch.optim as optim
import pandas as pd
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
import matplotlib.pyplot as plt 
import torch


def train_tft_model(train_dataloader, val_dataloader, training, best_params):
    """
    Initializes and trains a Temporal Fusion Transformer (TFT) model.

    This function takes the best hyperparameters found from an optimization process,
    builds the TFT model, and runs a training loop for a fixed number of epochs.
    It logs the training and validation loss at each epoch and saves the final
    loss history to a CSV file.

    Args:
        train_dataloader (DataLoader): DataLoader for the training set.
        val_dataloader (DataLoader): DataLoader for the validation set.
        training (TimeSeriesDataSet): The TimeSeriesDataSet object for training,
                                      containing necessary metadata for the model.
        best_params (dict): A dictionary containing the optimal hyperparameters
                            (e.g., learning_rate, dropout, hidden_size).

    Returns:
        TemporalFusionTransformer: The trained model object.
    """
    # --- 1. Setup Environment and Model Architecture ---
    # Set the computation device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Reconstruct model architecture details from the best hyperparameters
    hidden_size_multiple = best_params.get('hidden_size_multiple', 8) # Default if not in params
    attention_head_size = best_params.get('attention_head_size', 4)
    hidden_size = attention_head_size * hidden_size_multiple
    
    # Create a dictionary of parameters to pass to the model constructor
    params = {
        'hidden_size': hidden_size,
        'attention_head_size': best_params.get('attention_head_size', 4),
        'dropout': best_params.get('dropout', 0.1),
        'hidden_continuous_size': best_params.get('hidden_continuous_size', 8),
    }
    learning_rate = best_params.get('learning_rate', 1e-3)

    # Initialize the Temporal Fusion Transformer model from the dataset configuration
    tft_model = TemporalFusionTransformer.from_dataset(
        training,
        output_size=1, # We are predicting a single value (stock return)
        **params
    )
    tft_model.to(device) # Move the model to the selected device

    # --- 2. Define Loss Function and Optimizer ---
    loss_fn = MAE() # Use Mean Absolute Error as the loss metric
    print(f"Loss function: {type(loss_fn).__name__}")

    optimizer = optim.AdamW(tft_model.parameters(), lr=learning_rate)
    print(f"Optimizer: AdamW with lr={learning_rate}")

    # --- 3. Training Loop ---
    num_epochs = 200 
    print(f"Starting training for {num_epochs} epochs")

    train_losses_epoch = []
    val_losses_epoch = []

    # Loop over the dataset for the specified number of epochs
    for epoch in range(num_epochs):
        # --- Training Phase ---
        tft_model.train() # Set the model to training mode
        current_epoch_train_loss = 0.0
        train_batches_count = 0

        # Iterate over batches of data from the training dataloader
        for x_batch, y_batch_tuple in train_dataloader:
            # Move data batches to the computation device
            x_batch_device = {key: val.to(device) for key, val in x_batch.items()}
            y_true_device = y_batch_tuple[0].to(device)
            y_weights_device = y_batch_tuple[1].to(device) if len(y_batch_tuple) > 1 and y_batch_tuple[1] is not None else None
            y_batch_on_device_tuple = (y_true_device, y_weights_device)

            optimizer.zero_grad() # Clear previous gradients

            # Forward pass: compute predicted output by passing data to the model
            output_object = tft_model(x_batch_device)
            y_hat_predictions_tensor = output_object.prediction

            # Calculate the loss
            loss = loss_fn(y_hat_predictions_tensor, y_batch_on_device_tuple)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward() 
            # Update the model's weights
            optimizer.step() 

            current_epoch_train_loss += loss.item()
            train_batches_count += 1
        
        avg_epoch_train_loss = current_epoch_train_loss / train_batches_count
        train_losses_epoch.append(avg_epoch_train_loss)

        # --- Validation Phase ---
        tft_model.eval() # Set the model to evaluation mode
        current_epoch_val_loss = 0.0
        val_batches_count = 0
        with torch.no_grad(): # Disable gradient computation for validation
            for x_batch, y_batch_tuple in val_dataloader:
                x_batch_device = {key: val.to(device) for key, val in x_batch.items()}
                y_true_device = y_batch_tuple[0].to(device)
                y_weights_device = y_batch_tuple[1].to(device) if len(y_batch_tuple) > 1 and y_batch_tuple[1] is not None else None
                y_batch_on_device_tuple = (y_true_device, y_weights_device)

                output_object = tft_model(x_batch_device)
                y_hat_predictions_tensor = output_object.prediction
                val_loss = loss_fn(y_hat_predictions_tensor, y_batch_on_device_tuple)
                current_epoch_val_loss += val_loss.item()
                val_batches_count += 1
        
        avg_epoch_val_loss = current_epoch_val_loss / val_batches_count if val_batches_count > 0 else float('nan')
        val_losses_epoch.append(avg_epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_epoch_val_loss:.4f}")
    
    print("\nTraining finished.")

    # --- 4. Save Loss History ---
    # Create a DataFrame to store the loss history for later analysis
    loss_data = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Train_Loss': train_losses_epoch,
    'Validation_Loss': val_losses_epoch
    })
    loss_data.to_csv('training_losses.csv', index=False)
    
    return tft_model
