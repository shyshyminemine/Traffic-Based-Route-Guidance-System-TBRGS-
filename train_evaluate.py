import os
import sys

# Set working directory and module path to script directory for robust file loading
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Import from our custom modules
from data_preprocessing import prepare_dataloaders
from ml_models import build_lstm_model, build_gru_model, build_random_forest_model

import warnings
warnings.filterwarnings("ignore")

def evaluate_metrics(y_true, y_pred):
    """Calculate MSE, RMSE, NRMSE, and MAE metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / np.mean(y_true) if np.mean(y_true) != 0 else float('inf')
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, nrmse, mae

def plot_predictions(y_true, preds_dict, save_path="outputs/predictions_comparison.png"):
    """
    Plot actual vs predicted values for a subset of the test data (line chart).
    """
    plt.figure(figsize=(15, 6))
    
    plot_len = min(100, len(y_true))
    plt.plot(y_true[:plot_len], label="Actual Traffic", color='black', linewidth=2, marker='o', markersize=4)
    
    colors = ['blue', 'green', 'orange']
    for idx, (name, y_pred) in enumerate(preds_dict.items()):
        plt.plot(y_pred[:plot_len], label=f"{name} Pred", color=colors[idx], linestyle='--', marker='x', markersize=4)
        
    plt.title("Traffic Flow Prediction Comparison (First 100 Test Samples)")
    plt.xlabel("Time Step (15-min intervals)")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved prediction comparison plot to {save_path}")

def plot_loss_curves(histories_dict, save_path="outputs/loss_curves.png"):
    """
    Plot training and validation loss curves for each deep learning model.
    histories_dict: {model_name: keras History object}
    """
    n = len(histories_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, history) in zip(axes, histories_dict.items()):
        train_loss = history.history['loss']
        val_loss   = history.history['val_loss']
        epochs     = range(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, label='Train Loss', color='steelblue', linewidth=2)
        ax.plot(epochs, val_loss,   label='Val Loss',   color='tomato',    linewidth=2, linestyle='--')
        ax.set_title(f"{name} — Training vs Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved loss curves to {save_path}")


def plot_error_metrics(metrics_dict, save_path="outputs/metrics_comparison.png"):
    """
    Bar chart comparing MSE, RMSE, and MAE across all three models.
    """
    df_metrics = pd.DataFrame(metrics_dict).T
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    sns.set_theme(style="whitegrid")
    
    sns.barplot(x=df_metrics.index, y=df_metrics['MSE'], ax=axes[0], palette="viridis")
    axes[0].set_title("Mean Squared Error (MSE)")
    axes[0].set_ylabel("Error")
    
    sns.barplot(x=df_metrics.index, y=df_metrics['RMSE'], ax=axes[1], palette="plasma")
    axes[1].set_title("Root Mean Squared Error (RMSE)")
    axes[1].set_ylabel("Error")
    
    sns.barplot(x=df_metrics.index, y=df_metrics['NRMSE'], ax=axes[2], palette="mako")
    axes[2].set_title("Normalized RMSE (NRMSE)")
    axes[2].set_ylabel("Ratio")
    
    sns.barplot(x=df_metrics.index, y=df_metrics['MAE'], ax=axes[3], palette="magma")
    axes[3].set_title("Mean Absolute Error (MAE)")
    axes[3].set_ylabel("Error")
    
    for ax in axes:
        for container in ax.containers:
            # Format NRMSE slightly differently (4 decimals) vs others (2 decimals)
            fmt_str = '%.4f' if 'Ratio' in ax.get_ylabel() else '%.2f'
            ax.bar_label(container, fmt=fmt_str, padding=3, fontsize=10)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved error metrics comparison plot to {save_path}")


def main():
    FILE_PATH = "data/Scats Data October 2006.xls"
    WINDOW_SIZE = 4
    EPOCHS = 50
    BATCH_SIZE = 32
    
    print("1. Loading and preprocessing data...")
    _, _, _, scaler, arrays = prepare_dataloaders(FILE_PATH, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE)
    X_train, y_train, X_val, y_val, X_test, y_test = arrays
    
    predictions = {}
    metrics = {}
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # --- LSTM ---
    print("\n2. Training LSTM Model...")
    lstm_model = build_lstm_model(WINDOW_SIZE)
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )
    pred_lstm_scaled = lstm_model.predict(X_test, verbose=0).flatten()
    pred_lstm = scaler.inverse_transform(pred_lstm_scaled.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    mse_lstm, rmse_lstm, nrmse_lstm, mae_lstm = evaluate_metrics(y_test_inv, pred_lstm)
    metrics['LSTM'] = {'MSE': mse_lstm, 'RMSE': rmse_lstm, 'NRMSE': nrmse_lstm, 'MAE': mae_lstm}
    predictions['LSTM'] = pred_lstm
    print(f"LSTM   -> MSE: {mse_lstm:.2f}, RMSE: {rmse_lstm:.2f}, NRMSE: {nrmse_lstm:.4f}, MAE: {mae_lstm:.2f}")
    lstm_model.save("models/lstm_model.keras")

    # --- GRU ---
    print("\n3. Training GRU Model...")
    gru_model = build_gru_model(WINDOW_SIZE)
    gru_history = gru_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )
    pred_gru_scaled = gru_model.predict(X_test, verbose=0).flatten()
    pred_gru = scaler.inverse_transform(pred_gru_scaled.reshape(-1, 1)).flatten()
    
    mse_gru, rmse_gru, nrmse_gru, mae_gru = evaluate_metrics(y_test_inv, pred_gru)
    metrics['GRU'] = {'MSE': mse_gru, 'RMSE': rmse_gru, 'NRMSE': nrmse_gru, 'MAE': mae_gru}
    predictions['GRU'] = pred_gru
    print(f"GRU    -> MSE: {mse_gru:.2f}, RMSE: {rmse_gru:.2f}, NRMSE: {nrmse_gru:.4f}, MAE: {mae_gru:.2f}")
    gru_model.save("models/gru_model.keras")

    # --- Random Forest (Baseline) ---
    print("\n4. Training Random Forest Model (Baseline)...")
    rf_model = build_random_forest_model()
    # Random forest requires 2D input (samples, features) instead of 3D
    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_test_rf = X_test.reshape(X_test.shape[0], -1)
    
    rf_model.fit(X_train_rf, y_train)
    pred_rf_scaled = rf_model.predict(X_test_rf)
    pred_rf = scaler.inverse_transform(pred_rf_scaled.reshape(-1, 1)).flatten()
    
    mse_rf, rmse_rf, nrmse_rf, mae_rf = evaluate_metrics(y_test_inv, pred_rf)
    metrics['Random Forest'] = {'MSE': mse_rf, 'RMSE': rmse_rf, 'NRMSE': nrmse_rf, 'MAE': mae_rf}
    predictions['Random Forest'] = pred_rf
    print(f"RF     -> MSE: {mse_rf:.2f}, RMSE: {rmse_rf:.2f}, NRMSE: {nrmse_rf:.4f}, MAE: {mae_rf:.2f}")
    joblib.dump(rf_model, "models/random_forest_model.pkl")

    # --- Evaluate and Visualize ---
    print("\n5. Generating Evaluation Charts...")
    plot_predictions(y_test_inv, predictions)
    plot_error_metrics(metrics)
    plot_loss_curves({'LSTM': lstm_history, 'GRU': gru_history})
    
    # Determine the Best Model based on RMSE
    best_model_name = min(metrics, key=lambda k: metrics[k]['RMSE'])
    print("\n=======================================================")
    print(f" => BEST MODEL (lowest RMSE): {best_model_name}")
    print("=======================================================")

if __name__ == '__main__':
    main()
