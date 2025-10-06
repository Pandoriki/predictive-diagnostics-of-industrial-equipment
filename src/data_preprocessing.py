import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt
import os
import joblib

from configs.config import cfg

def load_data(data_dir: str = cfg.DATA_RAW_DIR) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                   [f'sensor_{i}' for i in range(1, 22)]

    df_train = pd.read_csv(os.path.join(data_dir, cfg.TRAIN_FILE), sep=' ', header=None)
    df_train.drop(columns=[26, 27], inplace=True) 
    df_train.columns = column_names

    df_test = pd.read_csv(os.path.join(data_dir, cfg.TEST_FILE), sep=' ', header=None)
    df_test.drop(columns=[26, 27], inplace=True) 
    df_test.columns = column_names

    df_rul_true = pd.read_csv(os.path.join(data_dir, cfg.RUL_TRUE_FILE), sep=' ', header=None)
    df_rul_true.drop(columns=[1], inplace=True) 
    df_rul_true.columns = ['RUL_true']

    return df_train, df_test, df_rul_true

def calculate_rul_for_train(df_train: pd.DataFrame, rul_cap: int = cfg.RUL_CAP) -> pd.DataFrame:
    max_time_per_unit = df_train.groupby('unit_number')['time_in_cycles'].max()
    df_train = df_train.merge(max_time_per_unit.rename('max_time_in_unit_cycles'), on='unit_number', how='left')
    df_train['RUL'] = df_train['max_time_in_unit_cycles'] - df_train['time_in_cycles']
    df_train.drop(columns=['max_time_in_unit_cycles'], inplace=True) 
    
    df_train['RUL'] = df_train['RUL'].clip(upper=rul_cap)
    
    return df_train

def preprocess_features(df_train: pd.DataFrame, df_test: pd.DataFrame,
                        fit_scaler: bool = True, scaler: MinMaxScaler = None,
                        scaler_save_path: str = None, features_save_path: str = None) \
                        -> (pd.DataFrame, pd.DataFrame, MinMaxScaler, list):
    
    irrelevant_sensor_cols = [f'sensor_{idx}' for idx in cfg.IRRELEVANT_SENSORS_INDICES]
    active_sensor_cols = [s for s in cfg.ALL_SENSOR_COLS if s not in irrelevant_sensor_cols]
    
    b, a = butter(N=3, Wn=0.05, btype='low', analog=False) 
    
    for sensor_idx in cfg.NOISY_SENSORS_FOR_FILTER_INDICES:
        sensor_name = f'sensor_{sensor_idx}'
        if sensor_name in active_sensor_cols:
            filtered_col_name = f'{sensor_name}_filtered'
            df_train[filtered_col_name] = df_train.groupby('unit_number')[sensor_name].transform(
                lambda x: filtfilt(b, a, x.values.astype(float))
            )
            df_test[filtered_col_name] = df_test.groupby('unit_number')[sensor_name].transform(
                lambda x: filtfilt(b, a, x.values.astype(float))
            )
            active_sensor_cols.remove(sensor_name)
            active_sensor_cols.append(filtered_col_name) 
    
    selected_features = cfg.OP_SETTING_COLS + active_sensor_cols
    
    if fit_scaler:
        scaler = MinMaxScaler()
        df_train[selected_features] = scaler.fit_transform(df_train[selected_features])
        if scaler_save_path:
            joblib.dump(scaler, scaler_save_path)
    else:
        if scaler is None:
            raise ValueError("Если fit_scaler=False, должен быть предоставлен обученный MinMaxScaler.")
        df_train[selected_features] = scaler.transform(df_train[selected_features])

    df_test[selected_features] = scaler.transform(df_test[selected_features])

    if features_save_path:
        joblib.dump(selected_features, features_save_path)
    
    return df_train, df_test, scaler, selected_features

def generate_sequences(df: pd.DataFrame, features: list, sequence_length: int = cfg.SEQUENCE_LENGTH) -> (np.ndarray, np.ndarray):
    X, y = [], []
    for unit_id in df['unit_number'].unique():
        subset = df[df['unit_number'] == unit_id]
        
        subset_rul = subset['RUL'].clip(upper=cfg.RUL_CAP)
        
        for i in range(len(subset) - sequence_length + 1):
            X.append(subset[features].iloc[i:i + sequence_length].values)
            y.append(subset_rul.iloc[i + sequence_length - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def generate_test_sequences_for_prediction(df_test: pd.DataFrame, features: list, sequence_length: int = cfg.SEQUENCE_LENGTH) -> np.ndarray:
    X_test_final = []
    for unit_id in df_test['unit_number'].unique():
        subset = df_test[df_test['unit_number'] == unit_id]
        
        if len(subset) >= sequence_length:
            X_test_final.append(subset[features].iloc[-sequence_length:].values)
        else:
            padded_sequence = np.zeros((sequence_length, len(features)), dtype=np.float32)
            actual_len = len(subset[features])
            padded_sequence[-actual_len:] = subset[features].values
            X_test_final.append(padded_sequence)
            
    return np.array(X_test_final, dtype=np.float32)

def get_rul_status(predicted_rul: float) -> (str, str, str):
    if predicted_rul > 14:
        return "НОРМАЛЬНО", "норма", "зеленый"
    elif predicted_rul > 7:
        return "ТРЕБУЕТ ОБСЛУЖИВАНИЯ", "предупреждение", "желтый"
    else:
        return "КРИТИЧЕСКОЕ СОСТОЯНИЕ", "критично", "красный"