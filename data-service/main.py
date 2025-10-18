import os
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

# Импорты конфигураций и ML-утилит
import configs.config as cfg
from src.data_preprocessing import load_data 
from src.predict_utils import get_rul_status 

# --- ГЛОБАЛЬНЫЕ РЕСУРСЫ ---
global_historical_data: Optional[pd.DataFrame] = None 
df_true_test_rul: Optional[pd.DataFrame] = None 


class HistoryResponse(BaseModel):
    unit_id: int = Field(..., description="ID оборудования.")
    time_in_cycles: List[int] = Field(..., description="Массив временных циклов, ось X для графиков.")
    rul_history: List[float] = Field(..., description="Массив значений RUL, ось Y для графика деградации.")
    sensor_data: Dict[str, List[float]] = Field(..., description="Словарь, где ключ - имя датчика, а значение - массив его показаний (ось Y).")

class EquipmentStatusSummary(BaseModel):
    unit_id: int
    current_rul: float
    status_ru: str
    status_code: Literal['normal', 'warning', 'critical'] 
    status_color: Literal['зеленый', 'желтый', 'красный'] 
    last_updated: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Data-Service APP: Запуск сервиса. Загрузка данных...")
    try:
        global global_historical_data, df_true_test_rul
        
        _, global_historical_data_raw, df_true_test_rul_temp = load_data(cfg.DATA_RAW_DIR)
        
        df_true_test_rul = df_true_test_rul_temp.copy()
        df_true_test_rul.columns = ['RUL_true']
        
        df_true_test_rul['unit_number'] = np.arange(1, len(df_true_test_rul) + 1)

        global_historical_data = global_historical_data_raw.copy()
        
        max_time_per_unit = global_historical_data.groupby('unit_number')['time_in_cycles'].max()
        global_historical_data = global_historical_data.merge(
            max_time_per_unit.rename('max_cycle_in_unit'), on='unit_number', how='left')

        print(f"[DATA_RESOURCES] Имитация исторической базы данных (test_FD001) загружена: {global_historical_data.shape}")
        print(f"[DATA_RESOURCES] DataFrame с истинными RUL исправлен и готов к работе.")
        
    except Exception as e:
        import traceback
        print(f"Data-Service APP: КРИТИЧЕСКАЯ ОШИБКА при старте: {e}")
        traceback.print_exc() 
        raise RuntimeError("Ошибка при инициализации данных. Сервис не может запуститься.")

    yield 
    print("Data-Service APP: Завершение работы сервиса.")


app = FastAPI(
    title="Предиктивная Диагностика: Data Service",
    description="Микросервис для доступа к имитации исторических данных оборудования.",
    version="1.1.1",
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
    if global_historical_data is not None and df_true_test_rul is not None:
        return {"status": "ok", "message": "Data-Service готов к работе."}
    else:
        raise HTTPException(status_code=503, detail="Data-Service инициализируется или имеет ошибки загрузки данных.")

@app.get("/equipment_list", response_model=List[int], tags=["Данные Оборудования"]) 
async def get_equipment_list_data():
    if global_historical_data is None:
        raise HTTPException(status_code=503, detail="Исторические данные не загружены.")
    return sorted(global_historical_data['unit_number'].unique().tolist())


@app.get("/history/{unit_id}", response_model=HistoryResponse, tags=["Данные Оборудования"]) 
async def get_unit_history_data(unit_id: int):
    global global_historical_data, df_true_test_rul

    if global_historical_data is None or df_true_test_rul is None:
        raise HTTPException(status_code=503, detail="Исторические данные не загружены.")

    unit_df = global_historical_data[global_historical_data['unit_number'] == unit_id]
    if unit_df.empty:
        raise HTTPException(status_code=404, detail=f"Исторические данные для оборудования ID {unit_id} не найдены.")

    time_in_cycles = unit_df['time_in_cycles'].tolist()

    true_rul_series = df_true_test_rul[df_true_test_rul['unit_number'] == unit_id]['RUL_true']
    true_rul_val = true_rul_series.iloc[0] if not true_rul_series.empty else float(cfg.RUL_CAP)
    
    last_cycle_in_test = unit_df['max_cycle_in_unit'].iloc[0]
    
    pseudo_rul_series = (true_rul_val + (last_cycle_in_test - unit_df['time_in_cycles']))
    rul_history = pseudo_rul_series.clip(lower=0, upper=cfg.RUL_CAP).tolist()

    sensor_data: Dict[str, List[float]] = {}
    for sensor_name in cfg.ALL_SENSOR_COLS:
        if sensor_name in unit_df.columns:
            sensor_data[sensor_name] = unit_df[sensor_name].tolist()
            
    return HistoryResponse(
        unit_id=unit_id,
        time_in_cycles=time_in_cycles,
        rul_history=rul_history,
        sensor_data=sensor_data
    )


@app.get("/status_summary", response_model=List[EquipmentStatusSummary], tags=["Обзор Статусов"])
async def get_all_equipment_status_summary_data():
    if global_historical_data is None or df_true_test_rul is None:
        raise HTTPException(status_code=503, detail="Исторические данные не загружены.")
    
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
    
    status_order = {'critical': 1, 'warning': 2, 'normal': 3}
    all_summaries.sort(key=lambda x: status_order[x.status_code])
    
    return all_summaries