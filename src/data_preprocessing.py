import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq 
import os
import joblib

import configs.config as cfg


def load_data(data_dir: str = None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    if data_dir is None:
        data_dir = cfg.DATA_RAW_DIR 

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

def calculate_rul_for_train(df_train: pd.DataFrame, rul_cap: int = None) -> pd.DataFrame:
    if rul_cap is None:
        rul_cap = cfg.RUL_CAP

    max_time_per_unit = df_train.groupby('unit_number')['time_in_cycles'].max()
    df_train = df_train.merge(max_time_per_unit.rename('max_cycle_in_unit'), on='unit_number', how='left')
    df_train['RUL'] = df_train['max_cycle_in_unit'] - df_train['time_in_cycles']
    df_train.drop(columns=['max_cycle_in_unit'], inplace=True) 
    
    df_train['RUL'] = df_train['RUL'].clip(upper=rul_cap)
    
    return df_train

def _apply_fft_features(df: pd.DataFrame, vibration_sensor_names: list, fft_bins: int) -> pd.DataFrame:
    if not vibration_sensor_names:
        return df

    all_units_with_fft = []
    
    for unit_id in df['unit_number'].unique():
        unit_subset = df[df['unit_number'] == unit_id].copy()
        
        unit_fft_aggregates = {} 
        
        for sensor_name in vibration_sensor_names:
            if sensor_name in unit_subset.columns:
                signal = unit_subset[sensor_name].values
                N = len(signal)
                
                if N == 0: continue
                
                yf = fft(signal)
                xf = fftfreq(N, 1)[:N//2]
                amplitudes = np.abs(yf[0:N//2])
                
                if xf.size == 0: continue
                
                bin_size = xf.max() / fft_bins
                
                for i in range(fft_bins):
                    low_freq = i * bin_size
                    high_freq = (i + 1) * bin_size
                    
                    bin_amplitudes = amplitudes[(xf >= low_freq) & (xf < high_freq)]
                    
                    mean_val = np.mean(bin_amplitudes) if len(bin_amplitudes) > 0 else 0.0
                    max_val = np.max(bin_amplitudes) if len(bin_amplitudes) > 0 else 0.0

                    unit_fft_aggregates[f'{sensor_name}_fft_bin{i}_mean'] = mean_val
                    unit_fft_aggregates[f'{sensor_name}_fft_bin{i}_max'] = max_val

        if not unit_fft_aggregates:
             all_units_with_fft.append(unit_subset)
             continue
        
        for feature_name, value in unit_fft_aggregates.items():
            unit_subset[feature_name] = value 

        all_units_with_fft.append(unit_subset)

    return pd.concat(all_units_with_fft, ignore_index=True)


def preprocess_features(df_train: pd.DataFrame, df_test: pd.DataFrame,
                        fit_scaler: bool = True, scaler: MinMaxScaler = None,
                        scaler_save_path: str = None, features_save_path: str = None,
                        rul_cap_val: int = None
                        ) -> (pd.DataFrame, pd.DataFrame, MinMaxScaler, list):
    
    if rul_cap_val is None:
        rul_cap_val = cfg.RUL_CAP

    irrelevant_sensor_cols = [f'sensor_{idx}' for idx in cfg.IRRELEVANT_SENSORS_INDICES]
    active_sensor_cols = [s for s in cfg.ALL_SENSOR_COLS if s not in irrelevant_sensor_cols]
    
    b, a = butter(N=3, Wn=0.05, btype='low', analog=False) 
    
    for sensor_idx in cfg.NOISY_SENSORS_FOR_FILTER_INDICES: # <- Здесь уже есть sensor_idx
        sensor_name = f'sensor_{sensor_idx}' # <- Используем sensor_idx, как определено в цикле
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
    
    fft_sensor_names_to_process = [f'sensor_{idx}' for idx in cfg.VIBRATION_SENSORS_FOR_FFT_INDICES]
    
    if cfg.USE_FFT_FEATURES:
        fft_source_sensors = [s for s in fft_sensor_names_to_process if s in df_train.columns]

        df_train = _apply_fft_features(df_train, fft_source_sensors, 
                                                    cfg.FFT_BINS_COUNT)
        df_test = _apply_fft_features(df_test, fft_source_sensors, 
                                                   cfg.FFT_BINS_COUNT)
        
        for sensor in fft_source_sensors:
            for i in range(cfg.FFT_BINS_COUNT):
                fft_col_mean = f'{sensor}_fft_bin{i}_mean'
                fft_col_max = f'{sensor}_fft_bin{i}_max'
                if fft_col_mean not in selected_features:
                    selected_features.append(fft_col_mean)
                if fft_col_max not in selected_features:
                    selected_features.append(fft_col_max)
        
        for feat in selected_features:
             if feat not in df_train.columns:
                 df_train[feat] = 0.0
             if feat not in df_test.columns:
                 df_test[feat] = 0.0
    
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


def generate_flat_features_for_boosting(df: pd.DataFrame, features: list, sequence_length: int = None) -> (np.ndarray, np.ndarray):
    if sequence_length is None:
        sequence_length = cfg.SEQUENCE_LENGTH

    X_flat, y_flat = [], []
    for unit_id in df['unit_number'].unique():
        subset = df[df['unit_number'] == unit_id]
        
        subset_rul = subset['RUL'].clip(upper=cfg.RUL_CAP)
        
        for i in range(len(subset) - sequence_length + 1):
            current_window_last_point = subset[features].iloc[i + sequence_length - 1].values 
            
            X_flat.append(current_window_last_point) 
            y_flat.append(subset_rul.iloc[i + sequence_length - 1])
            
    return np.array(X_flat, dtype=np.float32), np.array(y_flat, dtype=np.float32)


def generate_flat_test_features_for_boosting(df_test: pd.DataFrame, features: list, sequence_length: int = None) -> np.ndarray:
    if sequence_length is None:
        sequence_length = cfg.SEQUENCE_LENGTH

    X_test_flat_final = []
    for unit_id in df_test['unit_number'].unique():
        subset = df_test[df_test['unit_number'] == unit_id]
        
        if len(subset) >= sequence_length:
            current_window_last_point = subset[features].iloc[-1].values 
            X_test_flat_final.append(current_window_last_point)
        else:
            last_record_features = subset[features].iloc[-1].values if len(subset) > 0 else np.zeros(len(features), dtype=np.float32)
            X_test_flat_final.append(last_record_features)
            
    return np.array(X_test_flat_final, dtype=np.float32)