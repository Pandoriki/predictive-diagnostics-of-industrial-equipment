import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler 

import configs.config as cfg
from src.model_architecture import RULFilterNet
from src.data_preprocessing import calculate_rul_for_train, preprocess_features, load_data
from src.predict_utils import get_rul_status

app = FastAPI(
    title="ML RUL Prediction Service",
    description="API для предсказания Remaining Useful Life (RUL) промышленного оборудования.",
    version="1.0.0",
)

ml_model: Optional[RULFilterNet] = None
scaler: Optional[MinMaxScaler] = None
selected_features: Optional[List[str]] = None
device: Optional[torch.device] = None


global_historical_data: Optional[pd.DataFrame] = None
df_true_test_rul: Optional[pd.DataFrame] = None 

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
    """
    Запрос на предсказание RUL для одного агрегата.
    Содержит последовательность СЫРЫХ данных за SEQUENCE_LENGTH циклов.
    """
    unit_id: int = Field(..., description="Уникальный идентификатор оборудования")
    sequence_data: List[SensorDataPoint] = Field(..., min_items=cfg.SEQUENCE_LENGTH, max_items=cfg.SEQUENCE_LENGTH,
                                                 description=f"Последовательность сырых данных датчиков за {cfg.SEQUENCE_LENGTH} циклов.")

class PredictionResponse(BaseModel):
    """Ответ API с предсказанием RUL и статусом."""
    unit_id: int
    predicted_rul: float = Field(..., description="Прогнозируемое оставшееся время до отказа в циклах.")
    status_ru: str = Field(..., description="Статус оборудования на русском языке (НОРМАЛЬНО, ТРЕБУЕТ ОБСЛУЖИВАНИЯ, КРИТИЧЕСКОЕ СОСТОЯНИЕ).")
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

@app.on_event("startup")
async def load_ml_resources():
    global ml_model, scaler, selected_features, device, global_historical_data, df_true_test_rul

    device = torch.device(cfg.DEVICE)
    torch.manual_seed(cfg.RANDOM_SEED)

    model_path = os.path.join(cfg.MODELS_DIR, 'rul_prediction_model.pth')
    scaler_path = os.path.join(cfg.MODELS_DIR, 'scaler.pkl')
    features_path = os.path.join(cfg.MODELS_DIR, 'selected_features.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
        raise HTTPException(
            status_code=500,
            detail=f"Не найдены артефакты ML. Запустите 'python -m src.train_model'. Пути: {model_path}, {scaler_path}, {features_path}"
        )

    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler загружен из {scaler_path}")

        selected_features = joblib.load(features_path)
        print(f"Список признаков загружен из {features_path} ({len(selected_features)} признаков).")

        input_channels_count = len(selected_features)
        ml_model = RULFilterNet(input_channels=input_channels_count) 
        ml_model.load_state_dict(torch.load(model_path, map_location=device))
        ml_model.eval()
        ml_model.to(device)
        print(f"ML Модель '{type(ml_model).__name__}' загружена из {model_path} и готова на {device}.")

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

def preprocess_single_sequence_for_inference(raw_sequence_data: List[SensorDataPoint]) -> np.ndarray:
    """
    Принимает список 'сырых' SensorDataPoint за N циклов, применяет всю логику
    фильтрации и масштабирования (как при обучении) и формирует тензор для модели.
    """
    if scaler is None or selected_features is None:
        raise ValueError("ML ресурсы не загружены. Обратитесь к администратору.")

    df_raw_sequence = pd.DataFrame([s.model_dump() for s in raw_sequence_data])
    df_train_dummy = global_historical_data.head(1).copy() 

    all_raw_cols_for_processing = [c for c in df_raw_sequence.columns if c not in ['unit_number', 'time_in_cycles']]
    

    df_for_preprocess = df_raw_sequence[_full_raw_column_names].copy()

    _, df_processed_single_sequence, _, _ = \
        preprocess_features(df_train_dummy, df_for_preprocess.copy(), 
                            fit_scaler=False, scaler=scaler)

    final_sequence = df_processed_single_sequence[selected_features].values

    if final_sequence.shape != (cfg.SEQUENCE_LENGTH, len(selected_features)):
        raise ValueError("Ошибка формы данных после предобработки для инференса.")

    return final_sequence.astype(np.float32)


@app.post("/predict_rul", response_model=PredictionResponse)
async def predict_rul(request: PredictionRequest):
    """
    Принимает последовательность 'сырых' показаний датчиков для одного агрегата
    за `SEQUENCE_LENGTH` циклов и возвращает прогноз RUL и статус.
    """
    if ml_model is None or scaler is None or selected_features is None or device is None:
        raise HTTPException(status_code=503, detail="ML сервис не готов. Модель или скейлер не загружены.")

    try:
        processed_input_array = preprocess_single_sequence_for_inference(request.sequence_data)
        input_tensor = torch.tensor(processed_input_array, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_rul_tensor = ml_model(input_tensor)
            predicted_rul = predicted_rul_tensor.cpu().numpy().item() 

        status_ru, status_code, status_color = get_rul_status(predicted_rul)

        return PredictionResponse(
            unit_id=request.unit_id,
            predicted_rul=predicted_rul,
            status_ru=status_ru,
            status_code=status_code,
            status_color=status_color,
            reason=f"Прогноз RUL: {predicted_rul:.2f} циклов. "
                   f"Вероятно, причиной является изменение в {np.random.choice(selected_features, 1)[0]}."
                   # В реале - атрибуция причин, сейчас - заглушка
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Ошибка в данных запроса: {ve}. Проверьте формат.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при выполнении предсказания: {e}. Проверьте логи сервиса.")


@app.get("/history/{unit_id}", response_model=HistoryResponse)
async def get_unit_history(unit_id: int):
    """
    Возвращает исторические сырые данные по датчикам и псевдо-RUL для заданного unit_id.
    Использует кэшированные данные df_test (имитация БД для хакатона).
    """
    global global_historical_data, df_true_test_rul
    if global_historical_data is None or df_true_test_rul is None or selected_features is None:
        raise HTTPException(status_code=503, detail="Исторические данные или ML ресурсы не загружены.")

    unit_raw_data_full_df = global_historical_data[global_historical_data['unit_number'] == unit_id]
    if unit_raw_data_full_df.empty:
        raise HTTPException(status_code=404, detail=f"Исторические данные для оборудования ID {unit_id} не найдены.")

    last_cycle_of_unit_in_test_set = unit_raw_data_full_df['time_in_cycles'].max()
    unit_true_rul_val_from_file = df_true_test_rul[df_true_test_rul.index == unit_id-1]['RUL_true'].iloc[0]


    history_list: List[HistoryDataPointSimplified] = []
    
    df_temp_train_dummy = global_historical_data.head(1).copy() # Заглушка
    
    df_dummy_processed_no_scale_train, df_unit_data_for_history_filtered_only, _, _ = \
        preprocess_features(df_temp_train_dummy.copy(), unit_raw_data_full_df.copy(), 
                            fit_scaler=False, scaler=None, 
                            
                            )
    output_feature_names_for_history = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                                      [f'sensor_{i}' for i in range(1, 22) if f'sensor_{i}' not in cfg.IRRELEVANT_SENSORS_INDICES]

    for index, row in unit_raw_data_full_df.iterrows():
        current_cycle = row['time_in_cycles']

        pseudo_true_rul_for_display = (unit_true_rul_val_from_file + (last_cycle_of_unit_in_test_set - current_cycle)).clip(lower=0, upper=cfg.RUL_CAP)

        history_list.append(HistoryDataPointSimplified(
            time_in_cycles=current_cycle,
            true_rul_at_cycle=float(pseudo_true_rul_for_display),
            raw_feature_values=[float(row[f]) for f in output_feature_names_for_history],
        ))
        
    return HistoryResponse(unit_id=unit_id, history=history_list, feature_names=output_feature_names_for_history, original_feature_order=_full_raw_column_names)


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса."""
    if ml_model is not None and scaler is not None and selected_features is not None:
        return {"status": "ok", "message": "ML-сервис готов к работе"}
    else:
        raise HTTPException(status_code=503, detail="ML-сервис инициализируется или имеет ошибки.")


@app.get("/equipment_list", response_model=List[int])
async def get_equipment_list():
    """Возвращает список ID всех доступных тестовых агрегатов."""
    if global_historical_data is None:
        raise HTTPException(status_code=503, detail="Исторические данные не загружены.")
    
    return sorted(global_historical_data['unit_number'].unique().tolist())