from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

from backend.app.schemas.models import HistoryResponse, HistoryDataPointSimplified, EquipmentStatusSummary
from backend.app.core.ml_models import MLResources
from backend.app.core.config import settings
from src.predict_utils import get_rul_status
from src.data_preprocessing import preprocess_features 

router = APIRouter() # Создаем экземпляр роутера

# --- Вспомогательная функция для history endpoint ---
def _generate_history_for_unit(unit_id: int) -> Optional[List[HistoryDataPointSimplified]]:
    if MLResources.global_historical_data is None or MLResources.df_true_test_rul is None or MLResources.selected_features is None:
        raise ValueError("Исторические данные не загружены в MLResources.")

    unit_raw_data_full_df = MLResources.global_historical_data[MLResources.global_historical_data['unit_number'] == unit_id]
    if unit_raw_data_full_df.empty:
        return None
    
    true_rul_val_from_file_series = MLResources.df_true_test_rul[MLResources.df_true_test_rul['unit_number'] == unit_id]['RUL_true']
    if true_rul_val_from_file_series.empty:
         true_rul_val_from_file = float(settings.RUL_CAP)
    else:
         true_rul_val_from_file = true_rul_val_from_file_series.iloc[0]


    history_list: List[HistoryDataPointSimplified] = []
    
    # Список колонок для `raw_feature_values` (в `HistoryDataPointSimplified`)
    # Это должно быть согласовано с тем, что фронтенд ожидает отрисовать
    output_feature_names_for_history = []
    output_feature_names_for_history.extend(settings.OP_SETTING_COLS)
    for s_name in settings.ALL_SENSOR_COLS: 
        if s_name not in settings.IRRELEVANT_SENSORS_INDICES: # Только релевантные сырые датчики
            output_feature_names_for_history.append(s_name)
    
    # Добавляем фильтрованные датчики, если они есть
    for s_idx in settings.NOISY_SENSORS_FOR_FILTER_INDICES:
        s_name = f'sensor_{s_idx}'
        output_feature_names_for_history.append(f'{s_name}_filtered')

    # Добавляем FFT-признаки, если используются
    if settings.USE_FFT_FEATURES:
        for s_idx in settings.VIBRATION_SENSORS_FOR_FFT_INDICES:
            s_name = f'sensor_{s_idx}'
            for bin_idx in range(settings.FFT_BINS_COUNT):
                output_feature_names_for_history.append(f'{s_name}_fft_bin{bin_idx}_mean')
                output_feature_names_for_history.append(f'{s_name}_fft_bin{bin_idx}_max')

    
    for _, row in unit_raw_data_full_df.iterrows():
        current_cycle = row['time_in_cycles']
        last_cycle_of_unit_in_test_set = row['max_cycle_in_unit']

        pseudo_true_rul_for_display = (true_rul_val_from_file + (last_cycle_of_unit_in_test_set - current_cycle)).clip(lower=0, upper=settings.RUL_CAP)
        
        raw_values_for_display = []
        for feat_name in output_feature_names_for_history:
            # Используем .get() для безопасного доступа, если фичи нет
            raw_values_for_display.append(float(row.get(feat_name, 0.0))) 
        
        history_list.append(HistoryDataPointSimplified(
            time_in_cycles=current_cycle,
            true_rul_at_cycle=float(pseudo_true_rul_for_display),
            raw_feature_values=raw_values_for_display,
        ))
            
    return history_list

router.get("/history/{unit_id}", response_model=HistoryResponse, tags=["Данные Оборудования"]) 
async def get_unit_history(unit_id: int):
    history_data = _generate_history_for_unit(unit_id)
    if history_data is None:
        raise HTTPException(status_code=404, detail=f"Исторические данные для оборудования ID {unit_id} не найдены.")
    
    # Это для списка feature_names в HistoryResponse. 
    # Согласуется с output_feature_names_for_history
    output_feature_names_for_history_response = []
    output_feature_names_for_history_response.extend(settings.OP_SETTING_COLS)
    for s_name in settings.ALL_SENSOR_COLS: 
        if s_name not in settings.IRRELEVANT_SENSORS_INDICES:
            output_feature_names_for_history_response.append(s_name)
    for s_idx in settings.NOISY_SENSORS_FOR_FILTER_INDICES:
        s_name = f'sensor_{s_idx}'
        output_feature_names_for_history_response.append(f'{s_name}_filtered')
    if settings.USE_FFT_FEATURES:
        for s_idx in settings.VIBRATION_SENSORS_FOR_FFT_INDICES:
            s_name = f'sensor_{s_idx}'
            for bin_idx in range(settings.FFT_BINS_COUNT):
                output_feature_names_for_history_response.append(f'{s_name}_fft_bin{bin_idx}_mean')
                output_feature_names_for_history_response.append(f'{s_name}_fft_bin{bin_idx}_max')

    return HistoryResponse(unit_id=unit_id, history=history_data, 
                           feature_names=output_feature_names_for_history_response, 
                           original_feature_order=settings._meaningful_raw_columns)


router.get("/equipment_list", response_model=List[int], tags=["Данные Оборудования"]) 
async def get_equipment_list():
    if MLResources.global_historical_data is None:
        raise HTTPException(status_code=503, detail="Исторические данные не загружены.")
    
    return sorted(MLResources.global_historical_data['unit_number'].unique().tolist())


router.get("/status_summary", response_model=List[EquipmentStatusSummary], tags=["Обзор Статусов"])
async def get_all_equipment_status_summary():
    if MLResources.model is None or MLResources.scaler is None or MLResources.selected_features is None or MLResources.global_historical_data is None:
        raise HTTPException(status_code=503, detail="ML ресурсы не загружены для сводки статусов.")
    
    unique_units = MLResources.global_historical_data['unit_number'].unique().tolist()
    all_summaries: List[EquipmentStatusSummary] = []

    _full_raw_cols_for_status_summary_preprocess = settings._meaningful_raw_columns

    all_raw_sequences_dfs: List[pd.DataFrame] = []
    for unit_id in unique_units:
        unit_full_data = MLResources.global_historical_data[MLResources.global_historical_data['unit_number'] == unit_id]
        
        if len(unit_full_data) >= settings.SEQUENCE_LENGTH:
            last_n_cycles_df = unit_full_data.tail(settings.SEQUENCE_LENGTH).copy()
        else:
            padded_df = pd.DataFrame(np.zeros((settings.SEQUENCE_LENGTH, len(_full_raw_cols_for_status_summary_preprocess))), columns=_full_raw_cols_for_status_summary_preprocess)
            cols_to_fill = unit_full_data.columns.intersection(padded_df.columns)
            padded_df.iloc[-len(unit_full_data):][cols_to_fill] = unit_full_data[cols_to_fill].values
            last_n_cycles_df = padded_df


        all_raw_sequences_dfs.append(last_n_cycles_df)

    df_batch_raw_sequences = pd.concat(all_raw_sequences_dfs, ignore_index=True)
    
    df_train_dummy = pd.DataFrame(columns=_full_raw_cols_for_status_summary_preprocess + ['RUL', 'max_cycle_in_unit'])
    
    _, df_processed_batch, _, _ = preprocess_features(df_train_dummy.copy(), df_batch_raw_sequences.copy(),
                                                    fit_scaler=False, scaler=MLResources.scaler, is_single_unit_df=False)

    X_predict_ready = np.array([
        df_processed_batch[df_processed_batch['unit_number'] == uid][MLResources.selected_features].iloc[-1].values
        for uid in unique_units
    ])

    predictions = MLResources.model.predict(X_predict_ready)
    
    for i, unit_id in enumerate(unique_units):
        predicted_rul = predictions[i]
        status_ru, status_code, status_color = get_rul_status(predicted_rul)
        
        all_summaries.append(EquipmentStatusSummary(
            unit_id=int(unit_id),
            current_rul=float(predicted_rul),
            status_ru=status_ru,
            status_code=status_code,
            status_color=status_color,
            last_updated=datetime.now().strftime("%H:%M:%S")
        ))
    
    status_order = {'critical': 1, 'warning': 2, 'normal': 3}
    all_summaries.sort(key=lambda x: status_order[x.status_code])
    
    return all_summaries


router.get("/health", tags=["Утилиты"])
async def health_check():
    if MLResources.model is not None and MLResources.scaler is not None and MLResources.selected_features is not None:
        return {"status": "ok", "message": "Backend API готов к работе (ML загружен)."}
    else:
        raise HTTPException(status_code=503, detail="Backend API инициализируется или имеет ошибки загрузки ML.")