import os
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
# >>>>>> НОВОЕ: ИМПОРТ BASEMODEL И FIELD ИЗ PYDANTIC <<<<<<
from pydantic import BaseModel, Field
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from typing import List, Optional, Literal # Оставляем, так как используется

# Импорты конфигураций и ML-утилит
import configs.config as cfg
from src.data_preprocessing import load_data 
from src.predict_utils import get_rul_status 

# --- ГЛОБАЛЬНЫЕ РЕСУРСЫ (остаются, как были) ---
global_historical_data: Optional[pd.DataFrame] = None 
df_true_test_rul: Optional[pd.DataFrame] = None 


# --- Pydantic модели (теперь будут определены) ---
_full_raw_column_names = ['unit_number', 'time_in_cycles'] + cfg.OP_SETTING_COLS + cfg.ALL_SENSOR_COLS

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
    status_code: Literal['normal', 'warning', 'critical'] 
    status_color: Literal['зеленый', 'желтый', 'красный'] 
    last_updated: str

# --- Контекстный менеджер FastAPI для событий Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Data-Service APP: Запуск сервиса. Загрузка данных...")
    try:
        global global_historical_data, df_true_test_rul # Объявляем глобальные переменные
        
        # 1. Загрузка сырых тестовых данных для имитации БД
        _, global_historical_data_raw, df_true_test_rul_temp = load_data(cfg.DATA_RAW_DIR)
        
        global_historical_data = global_historical_data_raw.copy()
        df_true_test_rul = df_true_test_rul_temp
        
        # Добавляем max_cycle_in_unit к historical_data (это нужно для pseudo-RUL)
        max_time_per_unit = global_historical_data.groupby('unit_number')['time_in_cycles'].max()
        global_historical_data = global_historical_data.merge(
            max_time_per_unit.rename('max_cycle_in_unit'), on='unit_number', how='left')

        print(f"[DATA_RESOURCES] Имитация исторической базы данных (test_FD001) загружена: {global_historical_data.shape}")
        
    except Exception as e:
        import traceback
        print(f"Data-Service APP: КРИТИЧЕСКАЯ ОШИБКА при старте: {e}")
        traceback.print_exc() 
        raise RuntimeError("Ошибка при инициализации данных. Сервис не может запуститься.")

    yield 
    print("Data-Service APP: Завершение работы сервиса. Освобождение ресурсов (если нужно)...")


app = FastAPI(
    title="Предиктивная Диагностика: Data Service",
    description="Микросервис для доступа к имитации исторических данных оборудования.",
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


# --- API ЭНДПОИНТЫ (Data Service) ---

@app.get("/health", tags=["Утилиты"])
async def health_check_data():
    global global_historical_data, df_true_test_rul # Объявляем глобальные
    if global_historical_data is not None and df_true_test_rul is not None:
        return {"status": "ok", "message": "Data-Service готов к работе."}
    else:
        raise HTTPException(status_code=503, detail="Data-Service инициализируется или имеет ошибки загрузки данных.")

@app.get("/equipment_list", response_model=List[int], tags=["Данные Оборудования"]) 
async def get_equipment_list_data():
    global global_historical_data # Объявляем глобальную
    if global_historical_data is None:
        raise HTTPException(status_code=503, detail="Исторические данные не загружены.")
    
    return sorted(global_historical_data['unit_number'].unique().tolist())


@app.get("/history/{unit_id}", response_model=HistoryResponse, tags=["Данные Оборудования"]) 
async def get_unit_history_data(unit_id: int):
    global global_historical_data, df_true_test_rul # Объявляем глобальные

    if global_historical_data is None or df_true_test_rul is None:
        raise HTTPException(status_code=503, detail="Исторические данные не загружены.")

    unit_raw_data_full_df = global_historical_data[global_historical_data['unit_number'] == unit_id].copy()
    if unit_raw_data_full_df.empty:
        raise HTTPException(status_code=404, detail=f"Исторические данные для оборудования ID {unit_id} не найдены.")
    
    true_rul_val_from_file_series = df_true_test_rul[df_true_test_rul['unit_number'] == unit_id]['RUL_true']
    if true_rul_val_from_file_series.empty:
         true_rul_val_from_file = float(cfg.RUL_CAP)
    else:
         true_rul_val_from_file = true_rul_val_from_file_series.iloc[0]


    history_list: List[HistoryDataPointSimplified] = []
    
    output_feature_names_for_history = []
    output_feature_names_for_history.extend(cfg.OP_SETTING_COLS)
    for s_name in cfg.ALL_SENSOR_COLS: 
        if s_name not in cfg.IRRELEVANT_SENSORS_INDICES:
            output_feature_names_for_history.append(s_name)
    for s_idx in cfg.NOISY_SENSORS_FOR_FILTER_INDICES:
        s_name = f'sensor_{s_idx}'
        output_feature_names_for_history.append(f'{s_name}_filtered')
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
            raw_values_for_display.append(float(row.get(feat_name, 0.0))) 
        
        history_list.append(HistoryDataPointSimplified(
            time_in_cycles=current_cycle,
            true_rul_at_cycle=float(pseudo_true_rul_for_display),
            raw_feature_values=raw_values_for_display,
        ))
            
    return HistoryResponse(unit_id=unit_id, history=history_list, 
                           feature_names=output_feature_names_for_history, 
                           original_feature_order=cfg._meaningful_raw_columns)


@app.get("/status_summary", response_model=List[EquipmentStatusSummary], tags=["Обзор Статусов"])
async def get_all_equipment_status_summary_data():
    global global_historical_data, df_true_test_rul # Объявляем глобальные

    if global_historical_data is None or df_true_test_rul is None:
        raise HTTPException(status_code=503, detail="Исторические данные не загружены.")
    
    # Это mock-реализация, которая просто возвращает все единицы из исторических данных
    # и присваивает им случайные, но реалистичные RUL и статусы.
    unique_units = global_historical_data['unit_number'].unique().tolist()
    all_summaries: List[EquipmentStatusSummary] = []
    
    for unit_id in unique_units:
        rand_rul = float(np.random.rand() * cfg.RUL_CAP)
        status_ru, status_code, status_color = get_rul_status(rand_rul)
        
        all_summaries.append(EquipmentStatusSummary(
            unit_id=unit_id,
            current_rul=float(rand_rul),
            status_ru=status_ru,
            status_code=status_code,
            status_color=status_color,
            last_updated=datetime.now().strftime("%H:%M:%S")
        ))
    
    # Сортируем по критичности
    status_order = {'critical': 1, 'warning': 2, 'normal': 3}
    all_summaries.sort(key=lambda x: status_order[x.status_code])
    
    return all_summaries