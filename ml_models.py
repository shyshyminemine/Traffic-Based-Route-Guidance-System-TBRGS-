import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor

def build_lstm_model(window_size):
    """
    Build and compile an LSTM model.
    """
    model = Sequential([
        LSTM(64, input_shape=(window_size, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru_model(window_size):
    """
    Build and compile a GRU model.
    """
    model = Sequential([
        GRU(64, input_shape=(window_size, 1), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_random_forest_model():
    """
    Build a Random Forest Regressor model to serve as a strong ML baseline.
    Absorbing the idea from regression_tree_2B.ipynb: By flattening the 
    sliding window (samples, time_steps, 1) into (samples, features), 
    each historical time step acts as an autoregressive lag feature 
    (e.g., y_lag4, y_lag3, y_lag2, y_lag1) exactly as requested.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    return model
