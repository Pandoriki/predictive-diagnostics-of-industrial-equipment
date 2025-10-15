import os
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict # Добавлен Dict для pydantic_config

# Импортируем модули ML-логики, которые находятся на /app/configs/ и /app/src/
# Благодаря ENV PYTHONPATH /app:$PYTHONPATH в Dockerfile
import configs.config as cfg
from src.data_preprocessing import calculate_rul_for_train, preprocess_features, load_data
from src.predict_utils import get_rul_status

app = FastAPI(
    title="Предиктивная Диагностика: Unified Backend API",
    description="Объединенный API-сервис, включающий управление данными и ML-инференс для системы предиктивной диагностики.",
    version="1.0.0",
)

# --- MIDDLEWARE: CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)


# --- ГЛОБАЛЬНЫЕ РЕСУРСЫ ---
ml_model: Optional[CatBoostRegressor] = None 
scaler: Optional[MinMaxScaler] = None 
selected_features: Optional[List[str]] = None

# Имитация исторической базы данных (на основе test_FD001.txt)
global_historical_data_raw: Optional[pd.DataFrame] = None # Сырые тестовые данные
df_true_test_rul: Optional[pd.DataFrame] = None # RUL для последней точки тестовых юнитов
df_all_processed_history_data: Optional[pd.DataFrame] = None # Все данные test set, полностью предобработанные

# Полные имена колонок для SensorDataPoint (должны соответствовать df_test_raw)
_full_raw_column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                         [f'sensor_{i}' for i in range(1, 22)]


# --- Вспомогательные классы Pydantic (Согласованы с frontend/src/api/types.ts) ---
class SensorDataPoint(BaseModel):
    # Убираем __pydantic_config__ = {'extra': 'ignore'} здесь, 
    # лучше, если Pydantic будет требовать точные поля для входящего запроса.
    # Если данные с фронтенда могут быть неполными, тогда нужно добавлять.
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

class EquipmentStatusSummary(BaseModel):
    unit_id: int
    current_rul: float
    status_ru: str
    status_code: 'normal' | 'warning' | 'critical'
    status_color: 'зеленый' | 'желтый' | 'красный'
    last_updated: str


# --- Инициализация при старте FastAPI ---
@app.on_event("startup")
async def load_ml_and_data_resources():
    global ml_model, scaler, selected_features, global_historical_data_raw, \
           df_true_test_rul, df_all_processed_history_data

    model_path_cbm = os.path.join(cfg.MODELS_DIR, 'rul_prediction_model.cbm') 
    scaler_path = os.path.join(cfg.MODELS_DIR, 'scaler.pkl')
    features_path = os.path.join(cfg.MODELS_DIR, 'selected_features.pkl')

    if not os.path.exists(model_path_cbm) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
        raise HTTPException(
            status_code=500,
            detail=f"Не найдены артефакты ML. Запустите 'python -m src.train_model'. Проверены пути: {model_path_cbm}, {scaler_path}, {features_path}"
        )

    try:
        # Загрузка MinMaxScaler
        scaler = joblib.load(scaler_path)
        print(f"Scaler загружен из {scaler_path}")

        # Загрузка списка выбранных признаков
        selected_features = joblib.load(features_path)
        print(f"Список признаков загружен из {features_path} ({len(selected_features)} признаков).")

        # Загрузка CatBoost модели
        ml_model = CatBoostRegressor()
        ml_model.load_model(model_path_cbm)
        print(f"CatBoost модель загружена из {model_path_cbm}.")

        # Загрузка сырых тестовых данных для имитации истории/БД
        df_train_raw_dummy, global_historical_data_raw, df_true_test_rul = load_data(cfg.DATA_RAW_DIR) # df_train_raw_dummy - заглушка
        
        # Добавляем max_cycle_in_unit к сырым данным, для pseudo-RUL вычислений
        max_time_per_unit = global_historical_data_raw.groupby('unit_number')['time_in_cycles'].max()
        global_historical_data_raw = global_historical_data_raw.merge(
            max_time_per_unit.rename('max_cycle_in_unit'), on='unit_number', how='left')


        # Предобрабатываем ВЕСЬ тестовый набор, чтобы быстро отдавать уже обработанные фичи для History / StatusSummary
        # В preprocess_features требуется `df_train` для `fit_scaler=False`. Используем фиктивный, пустой DataFrame.
        df_empty_train = pd.DataFrame(columns=_full_raw_column_names + ['RUL', 'max_cycle_in_unit'])
        df_all_processed_history_data, _, _, _ = preprocess_features(
            df_empty_train, global_historical_data_raw.copy(), # !!! Pass a copy of raw data, not already preprocessed
            fit_scaler=False, scaler=scaler, rul_cap_val=cfg.RUL_CAP
        )
        print(f"Все исторические данные (обработанные) для сервисов загружены: {df_all_processed_history_data.shape}")
        
    except Exception as e:
        import traceback
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить ML ресурсы: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Сервис не готов: ошибка загрузки ML ресурсов. Проверьте логи.")


# --- Вспомогательная функция для предобработки одной последовательности для CatBoost инференса ---
def preprocess_single_sequence_for_inference(raw_sequence_data: List[SensorDataPoint]) -> np.ndarray:
    global scaler, selected_features # Access global variables

    if scaler is None or selected_features is None or ml_model is None:
        raise ValueError("ML ресурсы не загружены. Обратитесь к администратору.")

    df_raw_sequence = pd.DataFrame([s.model_dump() for s in raw_sequence_data])

    # Гарантируем наличие всех ожидаемых колонок в DF (заполняем нулями, если нет)
    for col in _full_raw_column_names:
        if col not in df_raw_sequence.columns:
            df_raw_sequence[col] = 0.0 # Заполняем 0 для отсутствующих сенсоров, чтобы избежать ошибок


    df_for_preprocess = df_raw_sequence[_full_raw_column_names].copy()
    
    df_train_dummy = pd.DataFrame(columns=_full_raw_column_names + ['RUL', 'max_cycle_in_unit']) 
    
    _, df_processed_single_sequence, _, _ = \
        preprocess_features(df_train_dummy.copy(), df_for_preprocess.copy(), 
                            fit_scaler=False, scaler=scaler)

    final_flat_features = df_processed_single_sequence[selected_features].iloc[-1].values 
    
    if final_flat_features.shape != (len(selected_features),):
        raise ValueError(f"Ошибка формы данных после предобработки для инференса CatBoost. Ожидаемо ({len(selected_features)},), получено {final_flat_features.shape}")

    return final_flat_features.astype(np.float32)


# --- API ЭНДПОИНТЫ ---

@app.post("/api/predict_rul", response_model=PredictionResponse, tags=["Прогноз RUL"])
async def predict_rul_api(request: PredictionRequest): # Changed function name to avoid conflict with method name
    if ml_model is None or scaler is None or selected_features is None:
        raise HTTPException(status_code=503, detail="ML сервис не готов. Модель или скейлер не загружены.")

    try:
        processed_input_array = preprocess_single_sequence_for_inference(request.sequence_data)
        predicted_rul = ml_model.predict(processed_input_array.reshape(1, -1)).item()
        
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


@app.get("/api/history/{unit_id}", response_model=HistoryResponse, tags=["Данные Оборудования"])
async def get_unit_history_api(unit_id: int): # Changed function name
    global global_historical_data_raw, df_true_test_rul, selected_features # Используем raw_data

    if global_historical_data_raw is None or df_true_test_rul is None or selected_features is None or df_all_processed_history_data is None:
        raise HTTPException(status_code=503, detail="Исторические данные или ML ресурсы не загружены.")

    # Используем сырые данные для получения оригинальных значений
    unit_raw_data_full_df = global_historical_data_raw[global_historical_data_raw['unit_number'] == unit_id]
    if unit_raw_data_full_df.empty:
        raise HTTPException(status_code=404, detail=f"Исторические данные для оборудования ID {unit_id} не найдены.")
    
    # Расчет pseudo_RUL
    true_rul_val_from_file_series = df_true_test_rul[df_true_test_rul['unit_number'] == unit_id]['RUL_true']
    if true_rul_val_from_file_series.empty:
         true_rul_val_from_file = float(cfg.RUL_CAP)
    else:
         true_rul_val_from_file = true_rul_val_from_file_series.iloc[0]


    history_list: List[HistoryDataPointSimplified] = []
    
    # Формируем список feature_names, которые отдаем (они должны быть в том же порядке, как raw_feature_values)
    output_feature_names_for_history = []
    output_feature_names_for_history.extend(cfg.OP_SETTING_COLS)
    for s_idx in range(1, 22):
        s_name = f'sensor_{s_idx}'
        if s_name not in cfg.IRRELEVANT_SENSORS_INDICES: 
            output_feature_names_for_history.append(s_name)
        # Добавляем фильтрованные сенсоры
        if s_idx in cfg.NOISY_SENSORS_FOR_FILTER_INDICES:
            output_feature_names_for_history.append(f'{s_name}_filtered')
    
    # Добавляем FFT признаки, если включены
    if cfg.USE_FFT_FEATURES:
        for s_idx in cfg.VIBRATION_SENSORS_FOR_FFT_INDICES:
            s_name = f'sensor_{s_idx}'
            for bin_idx in range(cfg.FFT_BINS_COUNT):
                output_feature_names_for_history.append(f'{s_name}_fft_bin{bin_idx}_mean')
                output_feature_names_for_history.append(f'{s_name}_fft_bin{bin_idx}_max')


    for _, row in unit_raw_data_full_df.iterrows():
        current_cycle = row['time_in_cycles']
        last_cycle_of_unit_in_test_set = row['max_cycle_in_unit']

        pseudo_true_rul_for_display = (true_rul_val_from_file + (last_cycle_of_unit_in_test_set - current_cycle)).clip(lower=0, upper=cfg.RUL_CAP)
        
        raw_values_for_display = []
        for feat_name in output_feature_names_for_history:
            # Убедитесь, что эта фича существует в `row` (для filtered/fft).
            # В `unit_raw_data_full_df` будут только исходные сырые колонки.
            # Фильтрованные и FFT-колонки появились после `preprocess_features`
            # Это значит, что для отдачи `filtered/fft` через этот эндпоинт,
            # мы должны отдавать не из `unit_raw_data_full_df`, а из `df_all_processed_history_data`.
            # Это важный момент!
            
            # --- ИСПРАВЛЕНИЕ: Берем фичи из df_all_processed_history_data, а не unit_raw_data_full_df ---
            processed_row = df_all_processed_history_data[(df_all_processed_history_data['unit_number'] == unit_id) & 
                                                          (df_all_processed_history_data['time_in_cycles'] == current_cycle)]
            if not processed_row.empty and feat_name in processed_row.columns:
                 raw_values_for_display.append(float(processed_row[feat_name].iloc[0])) #iloc[0] для извлечения скаляра
            else:
                 raw_values_for_display.append(0.0) # Заглушка, если нет данных (редко)
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---


        history_list.append(HistoryDataPointSimplified(
            time_in_cycles=current_cycle,
            true_rul_at_cycle=float(pseudo_true_rul_for_display),
            raw_feature_values=raw_values_for_display,
        ))
            
    return HistoryResponse(unit_id=unit_id, history=history_list, 
                           feature_names=output_feature_names_for_history, 
                           original_feature_order=_full_raw_column_names)


@app.get("/api/equipment_list", response_model=List[int], tags=["Данные Оборудования"])
async def get_equipment_list_api(): # Changed function name
    if global_historical_data_raw is None:
        raise HTTPException(status_code=503, detail="Исторические данные не загружены.")
    
    return sorted(global_historical_data_raw['unit_number'].unique().tolist())


@app.get("/api/status_summary", response_model=List[EquipmentStatusSummary], tags=["Обзор Статусов"])
async def get_all_equipment_status_summary_api(): # Changed function name
    if global_historical_data_raw is None or ml_model is None or scaler is None or selected_features is None or df_all_processed_history_data is None:
        raise HTTPException(status_code=503, detail="ML ресурсы не загружены для сводки статусов.")
    
    unique_units = df_all_processed_history_data['unit_number'].unique().tolist()
    all_summaries: List[EquipmentStatusSummary] = []

    # Готовим данные для массового инференса CatBoost
    # Берем последние срезы из полностью предобработанных данных (которые теперь включают фильтры и FFT)
    X_predict_ready_for_summary = generate_flat_test_features_for_boosting(df_all_processed_history_data, selected_features, cfg.SEQUENCE_LENGTH)

    if len(X_predict_ready_for_summary) != len(unique_units):
        print(f"Warning: Discrepancy in unit count for summary: unique_units={len(unique_units)}, X_predict_ready_for_summary={len(X_predict_ready_for_summary)}")

    predictions = ml_model.predict(X_predict_ready_for_summary) # Массовое предсказание CatBoost
    
    for i, unit_id in enumerate(unique_units):
        predicted_rul = predictions[i]
        status_ru, status_code, status_color = get_rul_status(predicted_rul)
        
        all_summaries.append(EquipmentStatusSummary(
            unit_id=int(unit_id),
            current_rul=float(predicted_rul),
            status_ru=status_ru,
            status_code=status_code,
            status_color=status_color,
            last_updated=pd.Timestamp.now().strftime("%H:%M:%S")
        ))
    
    status_order = {'critical': 1, 'warning': 2, 'normal': 3}
    all_summaries.sort(key=lambda x: status_order[x.status_code])
    
    return all_summaries

@app.get("/api/health", tags=["Утилиты"])
async def health_check():
    if ml_model is not None and scaler is not None and selected_features is not None:
        return {"status": "ok", "message": "Backend API готов к работе (ML загружен)."}
    else:
        raise HTTPException(status_code=503, detail="Backend API инициализируется или имеет ошибки загрузки ML.")