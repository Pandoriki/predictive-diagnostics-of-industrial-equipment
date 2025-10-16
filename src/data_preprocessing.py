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

    temp_col_names_for_txt_read = cfg._raw_data_full_column_names_from_txt 
    
    df_train = pd.read_csv(os.path.join(data_dir, cfg.TRAIN_FILE), sep=' ', header=None)
    df_train.columns = temp_col_names_for_txt_read 
    df_train.drop(columns=['_dummy_col_26', '_dummy_col_27'], inplace=True) 

    df_test = pd.read_csv(os.path.join(data_dir, cfg.TEST_FILE), sep=' ', header=None)
    df_test.columns = temp_col_names_for_txt_read 
    df_test.drop(columns=['_dummy_col_26', '_dummy_col_27'], inplace=True) 

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

def _apply_fft_features_single_unit(df_unit_data: pd.DataFrame, vibration_sensor_names: list, fft_bins: int) -> pd.DataFrame:
    if not vibration_sensor_names:
        return df_unit_data

    df_with_fft = df_unit_data.copy()

    for sensor_name in vibration_sensor_names:
        if sensor_name in df_unit_data.columns:
            signal = df_unit_data[sensor_name].values
            N = len(signal)
            
            if N < 2: 
                for i in range(fft_bins):
                    df_with_fft[f'{sensor_name}_fft_bin{i}_mean'] = 0.0
                    df_with_fft[f'{sensor_name}_fft_bin{i}_max'] = 0.0
                continue
            
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

                df_with_fft[f'{sensor_name}_fft_bin{i}_mean'] = mean_val
                df_with_fft[f'{sensor_name}_fft_bin{i}_max'] = max_val
    
    return df_with_fft


def _apply_fft_features_grouped(df: pd.DataFrame, vibration_sensor_names: list, fft_bins: int) -> pd.DataFrame:
    if not vibration_sensor_names:
        return df

    processed_dfs = []
    for unit_id, group in df.groupby('unit_number'):
        processed_group = _apply_fft_features_single_unit(group.copy(), vibration_sensor_names, fft_bins)
        processed_dfs.append(processed_group)
    return pd.concat(processed_dfs, ignore_index=True)


def preprocess_features(df_train: pd.DataFrame, df_test: pd.DataFrame, 
                        fit_scaler: bool = True, scaler: MinMaxScaler = None,
                        scaler_save_path: str = None, features_save_path: str = None,
                        is_single_unit_df: bool = False 
                        ) -> (pd.DataFrame, pd.DataFrame, MinMaxScaler, list):
    
    irrelevant_sensor_cols = [f'sensor_{idx}' for idx in cfg.IRRELEVANT_SENSORS_INDICES]
    active_sensor_cols_initial = [s for s in cfg.ALL_SENSOR_COLS if s not in irrelevant_sensor_cols] 
    
    b, a = butter(N=3, Wn=0.05, btype='low', analog=False) 

    MIN_LEN_FOR_FILTFILT = 15 

    def apply_filter_for_df(df_input: pd.DataFrame, sensor_name: str, filtered_col_name: str, apply_grouped: bool):
        if sensor_name not in df_input.columns:
            df_input[filtered_col_name] = 0.0
            return df_input

        signal_data = df_input[sensor_name].values.astype(float)
        
        if len(signal_data) < MIN_LEN_FOR_FILTFILT:
            df_input[filtered_col_name] = signal_data 
        elif apply_grouped and 'unit_number' in df_input.columns and df_input['unit_number'].nunique() > 1: 
            df_input[filtered_col_name] = df_input.groupby('unit_number')[sensor_name].transform(
                lambda x: filtfilt(b, a, x.values.astype(float)) if len(x) >= MIN_LEN_FOR_FILTFILT else x.values
            )
        else: 
            df_input[filtered_col_name] = filtfilt(b, a, signal_data)
        return df_input
            
    processed_noisy_sensor_names = [] 
    for sensor_idx in cfg.NOISY_SENSORS_FOR_FILTER_INDICES:
        sensor_name = f'sensor_{sensor_idx}' 
        if sensor_name in df_train.columns and sensor_name in df_test.columns: 
            filtered_col_name = f'{sensor_name}_filtered'
            df_train = apply_filter_for_df(df_train, sensor_name, filtered_col_name, apply_grouped=not is_single_unit_df)
            df_test = apply_filter_for_df(df_test, sensor_name, filtered_col_name, apply_grouped=not is_single_unit_df)
            
            processed_noisy_sensor_names.append(filtered_col_name)

    remaining_active_original_sensors = [s for s in active_sensor_cols_initial if s not in [f'sensor_{idx}' for idx in cfg.NOISY_SENSORS_FOR_FILTER_INDICES]]
    selected_features = list(cfg.OP_SETTING_COLS + remaining_active_original_sensors + processed_noisy_sensor_names)
    
    fft_source_sensors = [f'sensor_{idx}' for idx in cfg.VIBRATION_SENSORS_FOR_FFT_INDICES if f'sensor_{idx}' in df_train.columns] 
    
    if cfg.USE_FFT_FEATURES:
        
        if is_single_unit_df: 
            df_train = _apply_fft_features_single_unit(df_train, fft_source_sensors, cfg.FFT_BINS_COUNT)
            df_test = _apply_fft_features_single_unit(df_test, fft_source_sensors, cfg.FFT_BINS_COUNT)
        else: 
            df_train = _apply_fft_features_grouped(df_train, fft_source_sensors, cfg.FFT_BINS_COUNT)
            df_test = _apply_fft_features_grouped(df_test, fft_source_sensors, cfg.FFT_BINS_COUNT)
        
        for sensor in fft_source_sensors:
            for i in range(cfg.FFT_BINS_COUNT):
                selected_features.append(f'{sensor}_fft_bin{i}_mean')
                selected_features.append(f'{sensor}_fft_bin{i}_max')
        
        for df in [df_train, df_test]:
             for feat in selected_features:
                 if feat not in df.columns:
                     df[feat] = 0.0
    
    df_train.fillna(0.0, inplace=True) 
    df_test.fillna(0.0, inplace=True)
        
    if fit_scaler:
        scaler = MinMaxScaler()
        df_train_temp_for_scaler = df_train[selected_features].copy()
        df_train[selected_features] = scaler.fit_transform(df_train_temp_for_scaler)
        if scaler_save_path:
            joblib.dump(scaler, scaler_save_path)
    else:
        if scaler is None:
            raise ValueError("Если fit_scaler=False, должен быть предоставлен обученный MinMaxScaler.")
        df_test_temp_for_scaler = df_test[selected_features].copy() 
        df_test[selected_features] = scaler.transform(df_test_temp_for_scaler)

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
            last_record_features = subset[features].iloc[-1].values if not subset.empty else np.zeros(len(features), dtype=np.float32)
            X_test_flat_final.append(last_record_features)
            
    return np.array(X_test_flat_final, dtype=np.float32)