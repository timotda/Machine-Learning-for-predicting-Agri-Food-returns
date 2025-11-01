import torch
import torch.optim as optim
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
import optuna

def run_optuna_hpo(training, train_dataloader, val_dataloader, n_trials: int = 20):
    """
    Runs a hyperparameter optimization study using Optuna to find the best
    set of parameters for the Temporal Fusion Transformer model.

    Args:
        training (TimeSeriesDataSet): The TimeSeriesDataSet object for training,
                                      containing necessary metadata for the model.
        train_dataloader (DataLoader): DataLoader for the training set.
        val_dataloader (DataLoader): DataLoader for the validation set.
        n_trials (int): The total number of different hyperparameter combinations
                        (trials) to test.

    Returns:
        optuna.trial.FrozenTrial: The best trial object, which contains the
                                  best parameters and the final validation loss value.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Defines a single trial for the Optuna study.

        For each trial, Optuna suggests a set of hyperparameters. This function
        builds a model with these parameters, trains it for a few epochs,
        evaluates its performance on the validation set, and returns the
        validation loss.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object that provides methods
                                        for suggesting hyperparameter values.

        Returns:
            float: The best validation loss achieved during the trial.
        """
        # --- 1. Setup Environment and Suggest Hyperparameters ---
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))
        print(f"Training on device {device!r}.")

        # Optuna suggests a value for each hyperparameter from a defined range
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4])
        hidden_size_multiple = trial.suggest_int("hidden_size_multiple", 4, 16)
        hidden_size = attention_head_size * hidden_size_multiple

        max_hcs = max(hidden_size // 2, 4)
        hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 4, max_hcs)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)

        # --- 2. Build and Train Model for the Trial ---
        model = TemporalFusionTransformer.from_dataset(
            training,
            output_size=1,   
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
        )
        model.to(device)

        loss_fn = MAE()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Each trial is trained for a small number of epochs for efficiency
        hpo_num_epochs = 10
        best_val_loss_for_trial = float("inf")
        print("  Training model for trial")
        
        # --- 3. Short Training and Validation Loop ---
        for epoch in range(hpo_num_epochs):
            model.train()
            train_loss_accum = 0.0
            train_batches = 0
            for x_batch, y_batch_tuple in train_dataloader:
                x_device = {k: v.to(device) for k, v in x_batch.items()}
                y_true = y_batch_tuple[0].to(device)
                y_weights = (
                    y_batch_tuple[1].to(device)
                    if (len(y_batch_tuple) > 1 and y_batch_tuple[1] is not None)
                    else None
                )
                y_tuple = (y_true, y_weights)

                optimizer.zero_grad()
                output = model(x_device)
                y_pred = output.prediction
                loss = loss_fn(y_pred, y_tuple)
                loss.backward()
                optimizer.step()

                train_loss_accum += loss.item()
                train_batches += 1

            # --- Validation Phase for the Trial ---
            model.eval()
            val_loss_accum = 0.0
            val_batches = 0
            with torch.no_grad():
                for x_batch, y_batch_tuple in val_dataloader:
                    x_device = {k: v.to(device) for k, v in x_batch.items()}
                    y_true = y_batch_tuple[0].to(device)
                    y_weights = (
                        y_batch_tuple[1].to(device)
                        if (len(y_batch_tuple) > 1 and y_batch_tuple[1] is not None)
                        else None
                    )
                    y_tuple = (y_true, y_weights)

                    output = model(x_device)
                    y_pred = output.prediction
                    val_loss = loss_fn(y_pred, y_tuple)
                    val_loss_accum += val_loss.item()
                    val_batches += 1

            avg_val_loss = (
                val_loss_accum / val_batches if val_batches > 0 else float("inf")
            )
            print(f"    Epoch {epoch+1}/{hpo_num_epochs} -> Val Loss: {avg_val_loss:.4f}")
            best_val_loss_for_trial = min(best_val_loss_for_trial, avg_val_loss)

            # --- 4. Report to Optuna for Pruning ---
            # Report the validation loss to Optuna
            trial.report(avg_val_loss, epoch)
            # Check if the trial is unpromising and should be stopped early
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_loss_for_trial

    # --- 5. Create and Run the Optuna Study ---
    # Create the study with a median pruner to stop unpromising trials early
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    # Start the optimization process
    study.optimize(objective, n_trials=n_trials)

    return study.best_trial
