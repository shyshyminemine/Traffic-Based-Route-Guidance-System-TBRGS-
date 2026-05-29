import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

def load_and_clean_data(file_path, location=None):
    """
    Load Boroondara traffic data from Excel and clean missing values.
    Aligns data into a continuous time series with 15-minute intervals.
    """
    # Load dataset, header is at index 1
    df = pd.read_excel(file_path, sheet_name='Data', header=1)
    

    if location is None:
        location = df['Location'].unique()[0]
    
    df_loc = df[df['Location'] == location].copy()
    

    df_loc['Date'] = pd.to_datetime(df_loc['Date'])
    df_loc = df_loc.sort_values('Date').reset_index(drop=True)
    
    # Extract only the 15-minute interval columns (V00 to V95)
    traffic_cols = [f'V{str(i).zfill(2)}' for i in range(96)]
    
    # Melt V00-V95 columns into a continuous time series of 15-min intervals
    df_melted = df_loc.melt(id_vars=['Date'], value_vars=traffic_cols, 
                            var_name='TimeSlot', value_name='TrafficVolume')
    
    # Map 'V00' -> 0, 'V01' -> 1 to calculate exact timedelta
    df_melted['SlotIndex'] = df_melted['TimeSlot'].str.replace('V', '').astype(int)
    
    # Create an exact datetime column for each 15-minute interval
    df_melted['Datetime'] = df_melted['Date'] + pd.to_timedelta(df_melted['SlotIndex'] * 15, unit='m')
    

    df_melted = df_melted.sort_values('Datetime').reset_index(drop=True)
    
    # Handle missing values: ffill then bfill to cover leading gaps
    df_melted['TrafficVolume'] = df_melted['TrafficVolume'].ffill().bfill()
    
    return df_melted['TrafficVolume'].values.astype(float)


def create_sliding_window(series, window_size=4):
    """
    Create a sliding window dataset.
    window_size=4 means using past 1 hour (4 * 15 mins) to predict the next step.
    """
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def prepare_dataloaders(file_path, location=None, window_size=4, batch_size=32):
    """
    Complete pipeline: Load, clean, scale, window, split, and wrap into TF Datasets.
    """
    # 1. Load and clean data
    series = load_and_clean_data(file_path, location)
    
    # 2. Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    

    X, y = create_sliding_window(series_scaled, window_size)
    
    # Reshape X for LSTM/GRU expected input shape: (samples, time_steps, features)
    X = X.reshape(-1, window_size, 1)
    
    # 4. Split dataset: 70% Train, 15% Validation, 15% Test
    total_len = len(X)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.85)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # 5. Wrap into tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Also returning the raw arrays in case we need them directly (e.g., for scikit-learn evaluation metrics)
    arrays = (X_train, y_train, X_val, y_val, X_test, y_test)
    
    return train_dataset, val_dataset, test_dataset, scaler, arrays


if __name__ == "__main__":
    print("Starting Data Preprocessing Phase...")
    FILE_PATH = "data/Scats Data October 2006.xls"
    

    train_ds, val_ds, test_ds, scaler, arrays = prepare_dataloaders(
        file_path=FILE_PATH,
        window_size=4,   # Use past 4 time steps (1 hour)
        batch_size=32
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = arrays
    
    print("\n=== Data Preprocessing Summary ===")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape} (70%)")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape} (15%)")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape} (15%)")
    print("Data successfully wrapped into TensorFlow tf.data.Dataset objects!")
