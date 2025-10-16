import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from catboost import CatBoostRegressor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager 
from pydantic import BaseModel, Field
from typing import List, Optional, Literal 

from sklearn.preprocessing import MinMaxScaler 

# Импорты ML-логики (доступны благодаря PYTHONPATH=/app)
import configs.config as cfg
from src.data_preprocessing import preprocess_features 
from src.predict_utils import get_rul_status 


# --- ГЛОБАЛЬНЫЕ РЕСУРСЫ ---
ml_model: Optional[CatBoostRegressor] = None 
scaler: Optional[MinMaxScaler] = None 
selected_features: Optional[List[str]] = None


# --- Pydantic модели (копии, т.к. этот сервис автономен) ---
_full_raw_column_names = ['unit_number', 'time_in_cycles'] + cfg.OP_SETTING_COLS + cfg.ALL_SENSOR_COLS

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
    sequence_data: List[SensorDataPoint] = Field(..., description=f"Последовательность сырых данных датчиков за {cfg.SEQUENCE_LENGTH} циклов.")

class PredictionResponse(BaseModel):
    unit_id: int
    predicted_rul: float = Field(..., description="Прогнозируемое оставшееся время до отказа в циклах.")
    status_ru: str = Field(..., description="Статус оборудования на русском языке.")
    status_code: Literal['normal', 'warning', 'critical'] 
    status_color: Literal['зеленый', 'желтый', 'красный'] 
    reason: str = Field(..., description="Предполагаемая причина состояния (на основе правил, не ML).")

# (другие модели не нужны для ML-сервиса, т.к. он только предсказывает)

# --- Контекстный менеджер FastAPI для событий Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("ML-Service APP: Запуск сервиса. Загрузка ML ресурсов...")
    try:
        global ml_model, scaler, selected_features # Объявляем глобальные переменные
        
        model_path_cbm = os.path.join(cfg.MODELS_DIR, 'rul_prediction_model.cbm') 
        scaler_path = os.path.join(cfg.MODELS_DIR, 'scaler.pkl')
        features_path = os.path.join(cfg.MODELS_DIR, 'selected_features.pkl')

        if not os.path.exists(model_path_cbm) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
            raise RuntimeError(
                f"Не найдены артефакты ML для старта сервиса. Убедитесь, что 'train_model.py' был запущен. "
                f"Ожидаемые пути: {model_path_cbm}, {scaler_path}, {features_path}"
            )

        ml_model = CatBoostRegressor()
        ml_model.load_model(model_path_cbm)
        print(f"[ML_RESOURCES] CatBoost модель загружена из {model_path_cbm}")

        scaler = joblib.load(scaler_path)
        print(f"[ML_RESOURCES] Scaler загружен из {scaler_path}")

        selected_features = joblib.load(features_path)
        print(f"[ML_RESOURCES] Список признаков загружен из {features_path} ({len(selected_features)} признаков)")
        
    except Exception as e:
        import traceback
        print(f"ML-Service APP: КРИТИЧЕСКАЯ ОШИБКА при старте: {e}")
        traceback.print_exc() 
        raise RuntimeError("Ошибка при инициализации ML-ресурсов. Сервис не может запуститься.")

    yield 
    print("ML-Service APP: Завершение работы сервиса. Освобождение ресурсов (если нужно)...")


app = FastAPI(
    title="Предиктивная Диагностика: ML-Inference Service",
    description="Микросервис для выполнения ML-предсказаний RUL.",
    version="1.0.0",
    lifespan=lifespan 
)

# --- MIDDLEWARE: CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)


# --- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: Предобработка для инференса ---
def _preprocess_single_sequence_for_inference(raw_sequence_data: List[SensorDataPoint]) -> np.ndarray:
    """
    Выполняет полный пайплайн предобработки для одной последовательности
    сырых данных (от агрегата) перед подачей в CatBoost модель.
    """
    global ml_model, scaler, selected_features # Доступ к глобальным переменным

    if scaler is None or selected_features is None or ml_model is None: 
        raise ValueError("ML ресурсы не загружены.")

    df_raw_sequence = pd.DataFrame([s.model_dump() for s in raw_sequence_data])

    _full_raw_cols_for_processing = cfg._meaningful_raw_columns # Берем из конфига

    for col in _full_raw_cols_for_processing:
        if col not in df_raw_sequence.columns:
            df_raw_sequence[col] = 0.0 # Добавляем отсутствующие колонки как 0
    
    df_for_preprocess_actual = df_raw_sequence[_full_raw_cols_for_processing].copy()

    # Заглушка `df_train` для функции `preprocess_features`
    df_train_dummy_cols = _full_raw_cols_for_processing + ['RUL', 'max_cycle_in_unit']
    df_train_dummy = pd.DataFrame(columns=df_train_dummy_cols)
    
    # Вызов preprocess_features
    _, df_processed_single_sequence, _, _ = \
        preprocess_features(df_train=df_train_dummy.copy(), df_test=df_for_preprocess_actual.copy(), 
                            fit_scaler=False, scaler=scaler, is_single_unit_df=True)

    final_flat_features = df_processed_single_sequence[selected_features].iloc[-1].values 
    
    if final_flat_features.shape != (len(selected_features),):
        raise ValueError(f"Ошибка формы данных после предобработки для инференса CatBoost. Ожидаемо ({len(selected_features)},), получено {final_flat_features.shape}. Проверьте `selected_features`.")

    return final_flat_features.astype(np.float32)


# --- API ЭНДПОИНТЫ ---

@app.post("/predict_rul", response_model=PredictionResponse, tags=["Прогноз RUL"]) 
async def predict_rul_endpoint(request: PredictionRequest):
    global ml_model, scaler, selected_features # Объявляем глобальными (хотя тут используются)

    if ml_model is None or scaler is None or selected_features is None: 
        raise HTTPException(status_code=503, detail="ML сервис не готов. Модель или скейлер не загружены.")

    try:
        # Проверка длины входящей последовательности
        if len(request.sequence_data) != cfg.SEQUENCE_LENGTH: 
             raise HTTPException(status_code=400, detail=f"Ожидается последовательность данных из {cfg.SEQUENCE_LENGTH} циклов. Получено {len(request.sequence_data)}.")

        # Предобработка данных для инференса
        processed_input_array = _preprocess_single_sequence_for_inference(request.sequence_data)
        
        # Выполнение предсказания
        predicted_rul = ml_model.predict(processed_input_array.reshape(1, -1)).item()
        
        # Конвертация прогноза RUL в статус
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
        print(f"[{datetime.now().isoformat()}] Prediction Error: {e}")
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при выполнении предсказания: {e}. Проверьте логи сервиса.")


@app.get("/health", tags=["Утилиты"])
async def health_check():
    global ml_model, scaler, selected_features # Объявляем глобальными (хотя тут используются)

    if ml_model is not None and scaler is not None and selected_features is not None:
        return {"status": "ok", "message": "ML-Inference сервис готов к работе (ML загружен)."}
    else:
        raise HTTPException(status_code=503, detail="ML-Inference сервис инициализируется или имеет ошибки загрузки ML.")