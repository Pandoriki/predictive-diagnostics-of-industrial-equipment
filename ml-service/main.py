# ml-service/main.py
import os
import json
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor # !!! НОВОЕ

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# Загружаем cfg как модуль
import configs.config as cfg
# import src.model_architecture as model_arch # Больше не нужно
from src.data_preprocessing import calculate_rul_for_train, preprocess_features, load_data, generate_flat_test_features_for_boosting # Уточненные функции
from src.predict_utils import get_rul_status

app = FastAPI(
    title="Предиктивная Диагностика: ML-Сервис",
    description="API для предсказания Remaining Useful Life (RUL) промышленного оборудования.",
    version="1.0.0",
)

# Глобальные переменные для загруженных ресурсов ML
ml_model: Optional[CatBoostRegressor] = None # <<< Изменен тип модели
scaler: Optional[MinMaxScaler] = None
selected_features: Optional[List[str]] = None
# device: Optional[torch.device] = None # Больше не нужен для CatBoost

# Имитация исторических данных (для эндпоинта /history/{unit_id})
global_historical_data: Optional[pd.DataFrame] = None
df_true_test_rul: Optional[pd.DataFrame] = None

# --- Вспомогательные классы Pydantic (БЕЗ ИЗМЕНЕНИЙ) ---
_full_raw_column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                         [f'sensor_{i}' for i in range(1, 22)]

class SensorDataPoint(BaseModel):
    __pydantic_config__ = {'extra': 'ignore'}

    unit_number: int
    time_in_cycles: int
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

class PredictionRequest(BaseModel):
    unit_id: int = Field(..., description="Уникальный идентификатор оборудования")
    sequence_data: List[SensorDataPoint] = Field(..., min_items=cfg.SEQUENCE_LENGTH, max_items=cfg.SEQUENCE_LENGTH,
                                                 description=f"Последовательность сырых данных датчиков за {cfg.SEQUENCE_LENGTH} циклов.")

class PredictionResponse(BaseModel):
    unit_id: int
    predicted_rul: float = Field(..., description="Прогнозируемое оставшееся время до отказа в циклах.")
    status_ru: str = Field(..., description="Статус оборудования на русском языке.")
    status_code: str = Field(..., description="Краткий код статуса (normal, warning, critical).")
    status_color: str = Field(..., description="Предполагаемый цвет статуса (зеленый, желтый, красный).")
    reason: str = Field(..., description="Предполагаемая причина состояния (на основе правил, не ML).")

class HistoryDataPointSimplified(BaseModel):
    time_in_cycles: int
    true_rul_at_cycle: float = Field(..., description="Псевдо-RUL для отображения деградации на графике.")
    raw_feature_values: List[float] = Field(..., description="Исходные (немасштабированные) значения отобранных датчиков и опер. настроек.")

class HistoryResponse(BaseModel):
    unit_id: int
    history: List[HistoryDataPointSimplified]
    feature_names: List[str] 
    original_feature_order: List[str]

# --- Загрузка модели и скейлера при старте FastAPI ---
@app.on_event("startup")
async def load_ml_resources():
    global ml_model, scaler, selected_features, global_historical_data, df_true_test_rul

    # device = torch.device(cfg.DEVICE) # Удален
    # torch.manual_seed(cfg.RANDOM_SEED) # Удален
    
    model_path_cbm = os.path.join(cfg.MODELS_DIR, 'rul_prediction_model.cbm') # <<< НОВЫЙ ПУТЬ .cbm
    scaler_path = os.path.join(cfg.MODELS_DIR, 'scaler.pkl')
    features_path = os.path.join(cfg.MODELS_DIR, 'selected_features.pkl')

    if not os.path.exists(model_path_cbm) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
        raise HTTPException(
            status_code=500,
            detail=f"Не найдены артефакты ML. Запустите 'python -m src.train_model'. Пути: {model_path_cbm}, {scaler_path}, {features_path}"
        )

    try:
        # 1. Загрузка MinMaxScaler
        scaler = joblib.load(scaler_path)
        print(f"Scaler загружен из {scaler_path}")

        # 2. Загрузка списка выбранных признаков
        selected_features = joblib.load(features_path)
        print(f"Список признаков загружен из {features_path} ({len(selected_features)} признаков).")

        # 3. Загрузка CatBoost модели
        ml_model = CatBoostRegressor() # Инициализируем без параметров, они в save_model
        ml_model.load_model(model_path_cbm) # <<< НОВОЕ
        print(f"CatBoost модель загружена из {model_path_cbm}.")

        # 4. Загрузка сырых тестовых данных для имитации истории/БД
        _, global_historical_data_raw, df_true_test_rul_temp = load_data(cfg.DATA_RAW_DIR)
        
        global_historical_data = global_historical_data_raw.copy()
        df_true_test_rul = df_true_test_rul_temp
        
        max_time_per_unit = global_historical_data.groupby('unit_number')['time_in_cycles'].max()
        global_historical_data = global_historical_data.merge(
            max_time_per_unit.rename('max_cycle_in_unit'), on='unit_number', how='left')

        print(f"Имитация исторической базы данных загружена: {global_historical_data.shape}")
        
    except Exception as e:
        import traceback
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить ML ресурсы: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Сервис не готов: ошибка загрузки ML ресурсов. Проверьте логи.")


# --- Вспомогательная функция для предобработки одной последовательности (полный pipeline для инференса CatBoost) ---
def preprocess_single_sequence_for_inference(raw_sequence_data: List[SensorDataPoint]) -> np.ndarray:
    if scaler is None or selected_features is None or ml_model is None: # Проверим ml_model
        raise ValueError("ML ресурсы не загружены. Обратитесь к администратору.")

    df_raw_sequence = pd.DataFrame([s.model_dump() for s in raw_sequence_data])

    df_for_preprocess = df_raw_sequence[_full_raw_column_names].copy()

    # Для preprocess_features нужна df_train_dummy для fit_scaler=False
    df_train_dummy = pd.DataFrame(columns=_full_raw_column_names + ['RUL']) # Пустой DF с RUL колонкой
    
    # Применяем фильтрацию и масштабирование
    # (Функция preprocess_features теперь работает только на feature_engineering/scaling)
    # `df_train_dummy` просто placeholder, чтобы функция работала.
    _, df_processed_single_sequence, _, _ = \
        preprocess_features(df_train_dummy.copy(), df_for_preprocess.copy(), 
                            fit_scaler=False, scaler=scaler)

    # Извлекаем финальный вектор признаков. Для CatBoost это плоский вектор.
    final_flat_features = df_processed_single_sequence[selected_features].iloc[-1].values 
    
    # Проверка формы: (num_selected_features,)
    if final_flat_features.shape != (len(selected_features),):
        raise ValueError(f"Ошибка формы данных после предобработки для инференса CatBoost. Ожидаемо ({len(selected_features)},), получено {final_flat_features.shape}")

    return final_flat_features.astype(np.float32)


# --- API Endpoints ---
@app.post("/predict_rul", response_model=PredictionResponse)
async def predict_rul(request: PredictionRequest):
    if ml_model is None or scaler is None or selected_features is None:
        raise HTTPException(status_code=503, detail="ML сервис не готов. Модель или скейлер не загружены.")

    try:
        # Предобработка входных данных с полным пайплайном
        processed_input_array = preprocess_single_sequence_for_inference(request.sequence_data)

        # CatBoost предсказывает на 1D NumPy массивах (передаем как батч из 1 примера)
        predicted_rul = ml_model.predict(processed_input_array.reshape(1, -1)).item() # item() для извлечения скаляра
        
        # Применяем функцию постобработки для определения статуса
        status_ru, status_code, status_color = get_rul_status(predicted_rul)

        return PredictionResponse(
            unit_id=request.unit_id,
            predicted_rul=predicted_rul,
            status_ru=status_ru,
            status_code=status_code,
            status_color=status_color,
            reason=f"Прогноз RUL: {predicted_rul:.2f} циклов. Модель CatBoost." 
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Ошибка в данных запроса: {ve}. Проверьте формат.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при выполнении предсказания: {e}. Проверьте логи сервиса.")


@app.get("/history/{unit_id}", response_model=HistoryResponse)
async def get_unit_history(unit_id: int):
    global global_historical_data, df_true_test_rul
    if global_historical_data is None or df_true_test_rul is None or selected_features is None:
        raise HTTPException(status_code=503, detail="Исторические данные или ML ресурсы не загружены.")

    unit_raw_data_full_df = global_historical_data[global_historical_data['unit_number'] == unit_id]
    if unit_raw_data_full_df.empty:
        raise HTTPException(status_code=404, detail=f"Исторические данные для оборудования ID {unit_id} не найдены.")
    
    last_cycle_of_unit_in_test_set = unit_raw_data_full_df['time_in_cycles'].max()
    unit_true_rul_val_from_file = df_true_test_rul[df_true_test_rul['unit_number'] == unit_id]['RUL_true'].iloc[0] # Исправлено

    history_list: List[HistoryDataPointSimplified] = []
    
    output_feature_names_for_history = []
    # Колонок для отображения истории
    output_feature_names_for_history.extend(cfg.OP_SETTING_COLS)
    for s_idx in range(1,22):
        s_name = f'sensor_{s_idx}'
        if s_name not in cfg.IRRELEVANT_SENSORS_INDICES: # только релевантные сенсоры
            output_feature_names_for_history.append(s_name)
            # Если фильтровали, добавляем отфильтрованные тоже, иначе их не будет.
            if s_idx in cfg.NOISY_SENSORS_FOR_FILTER_INDICES:
                output_feature_names_for_history.append(f'{s_name}_filtered')
    # Если используем FFT, они тоже должны быть добавлены
    if cfg.USE_FFT_FEATURES:
        for s_idx in cfg.VIBRATION_SENSORS_FOR_FFT_INDICES:
            s_name = f'sensor_{s_idx}'
            for bin_idx in range(cfg.FFT_BINS_COUNT):
                output_feature_names_for_history.append(f'{s_name}_fft_bin{bin_idx}_mean')
                output_feature_names_for_history.append(f'{s_name}_fft_bin{bin_idx}_max')


    for _, row in unit_raw_data_full_df.iterrows():
        current_cycle = row['time_in_cycles']
        
        pseudo_true_rul_for_display = (unit_true_rul_val_from_file + (last_cycle_of_unit_in_test_set - current_cycle)).clip(lower=0, upper=cfg.RUL_CAP)

        # Берем только те фичи, которые будем отдавать
        raw_values = [float(row[f]) if f in row else 0.0 for f in output_feature_names_for_history]
        
        history_list.append(HistoryDataPointSimplified(
            time_in_cycles=current_cycle,
            true_rul_at_cycle=float(pseudo_true_rul_for_display),
            raw_feature_values=raw_values,
        ))
        
    return HistoryResponse(unit_id=unit_id, history=history_list, 
                           feature_names=output_feature_names_for_history, 
                           original_feature_order=_full_raw_column_names)


@app.get("/health")
async def health_check():
    if ml_model is not None and scaler is not None and selected_features is not None:
        return {"status": "ok", "message": "ML-сервис готов к работе"}
    else:
        raise HTTPException(status_code=503, detail="ML-сервис инициализируется или имеет ошибки.")


@app.get("/equipment_list", response_model=List[int])
async def get_equipment_list():
    if global_historical_data is None:
        raise HTTPException(status_code=503, detail="Исторические данные не загружены.")
    
    return sorted(global_historical_data['unit_number'].unique().tolist())