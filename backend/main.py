import os
import numpy as np
import pandas as pd
from datetime import datetime
import json

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager 
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict # <-- Добавлен Dict
import httpx

# Импорты конфигураций
import configs.config as cfg

# --- >>>>>>>>>>>> ГЛАВНОЕ ИСПРАВЛЕНИЕ ЗДЕСЬ <<<<<<<<<<<< ---
# Обновляем Pydantic модели, чтобы они соответствовали data-service

# УДАЛЯЕМ СТАРЫЕ МОДЕЛИ
# class HistoryDataPointSimplified(BaseModel):
#     ...
# class HistoryResponse(BaseModel):
#     unit_id: int
#     history: List[HistoryDataPointSimplified]
#     feature_names: List[str] 
#     original_feature_order: List[str]

# ДОБАВЛЯЕМ НОВУЮ, ПРАВИЛЬНУЮ МОДЕЛЬ
class HistoryResponse(BaseModel):
    unit_id: int = Field(..., description="ID оборудования.")
    time_in_cycles: List[int] = Field(..., description="Массив временных циклов, ось X для графиков.")
    rul_history: List[float] = Field(..., description="Массив значений RUL, ось Y для графика деградации.")
    sensor_data: Dict[str, List[float]] = Field(..., description="Словарь, где ключ - имя датчика, а значение - массив его показаний (ось Y).")


# Остальные модели (копируем их из data-service для консистентности)
class EquipmentStatusSummary(BaseModel):
    unit_id: int
    current_rul: float
    status_ru: str
    status_code: Literal['normal', 'warning', 'critical'] 
    status_color: Literal['зеленый', 'желтый', 'красный'] 
    last_updated: str

class SensorDataPoint(BaseModel):
    __pydantic_config__ = {'extra': 'ignore'}
    unit_number: int; time_in_cycles: int; op_setting_1: float; op_setting_2: float; op_setting_3: float
    sensor_1: float; sensor_2: float; sensor_3: float; sensor_4: float; sensor_5: float
    sensor_6: float; sensor_7: float; sensor_8: float; sensor_9: float; sensor_10: float
    sensor_11: float; sensor_12: float; sensor_13: float; sensor_14: float; sensor_15: float
    sensor_16: float; sensor_17: float; sensor_18: float; sensor_19: float; sensor_20: float; sensor_21: float

class PredictionRequest(BaseModel):
    unit_id: int
    sequence_data: List[SensorDataPoint]

class PredictionResponse(BaseModel):
    unit_id: int
    predicted_rul: float
    status_ru: str
    status_code: Literal['normal', 'warning', 'critical']
    status_color: Literal['зеленый', 'желтый', 'красный']
    reason: str
# --- >>>>>>>>>>>> КОНЕЦ ИСПРАВЛЕНИЯ <<<<<<<<<<<< ---


# --- GLOBAL HTTPX CLIENT ---
httpx_client: Optional[httpx.AsyncClient] = None

# --- Контекстный менеджер FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("API Gateway: Запуск сервиса...")
    global httpx_client
    httpx_client = httpx.AsyncClient()
    yield 
    print("API Gateway: Завершение работы сервиса. Закрытие HTTP клиента...")
    await httpx_client.aclose()


app = FastAPI(
    title="Предиктивная Диагностика: API Gateway",
    description="Основной API-шлюз, маршрутизирующий запросы к ML и Data-микросервисам.",
    version="1.1.0", # Версия обновлена
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/redoc"
)

# --- MIDDLEWARE: CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)


# --- API ENDPOINTS (остаются без изменений) ---

@app.post("/api/predict_rul", response_model=PredictionResponse, tags=["Прогноз RUL"]) 
async def predict_rul_endpoint(request: PredictionRequest):
    if httpx_client is None: raise HTTPException(status_code=500, detail="HTTP клиент не инициализирован.")
    try:
        ml_response = await httpx_client.post(f"{cfg.ML_SERVICE_URL}/predict_rul", json=request.model_dump())
        ml_response.raise_for_status()
        return PredictionResponse(**ml_response.json())
    except httpx.HTTPStatusError as e: raise HTTPException(status_code=e.response.status_code, detail=f"ML Service Error: {e.response.text}")
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Невозможно подключиться к ML-сервису: {e}")
    except Exception as e:
        import traceback; traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Ошибка Gateway при перенаправлении прогноза: {e}")


@app.get("/api/history/{unit_id}", response_model=HistoryResponse, tags=["Данные Оборудования"]) 
async def get_unit_history(unit_id: int):
    if httpx_client is None: raise HTTPException(status_code=500, detail="HTTP клиент не инициализирован.")
    try:
        data_response = await httpx_client.get(f"{cfg.DATA_SERVICE_URL}/history/{unit_id}")
        data_response.raise_for_status()
        # Теперь валидация пройдет успешно, так как модели совпадают
        return HistoryResponse(**data_response.json())
    except httpx.HTTPStatusError as e: raise HTTPException(status_code=e.response.status_code, detail=f"Data Service Error: {e.response.text}")
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Невозможно подключиться к Data-сервису: {e}")
    except Exception as e:
        import traceback; traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Ошибка Gateway при перенаправлении истории: {e}")


@app.get("/api/equipment_list", response_model=List[int], tags=["Данные Оборудования"]) 
async def get_equipment_list():
    if httpx_client is None: raise HTTPException(status_code=500, detail="HTTP клиент не инициализирован.")
    try:
        data_response = await httpx_client.get(f"{cfg.DATA_SERVICE_URL}/equipment_list")
        data_response.raise_for_status()
        return data_response.json()
    except httpx.HTTPStatusError as e: raise HTTPException(status_code=e.response.status_code, detail=f"Data Service Error: {e.response.text}")
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Невозможно подключиться к Data-сервису: {e}")
    except Exception as e:
        import traceback; traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Ошибка Gateway при получении списка: {e}")


@app.get("/api/status_summary", response_model=List[EquipmentStatusSummary], tags=["Обзор Статусов"])
async def get_all_equipment_status_summary():
    if httpx_client is None: raise HTTPException(status_code=500, detail="HTTP клиент не инициализирован.")
    try:
        data_response = await httpx_client.get(f"{cfg.DATA_SERVICE_URL}/status_summary")
        data_response.raise_for_status()
        return data_response.json()
    except httpx.HTTPStatusError as e: raise HTTPException(status_code=e.response.status_code, detail=f"Data Service Error: {e.response.text}")
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Невозможно подключиться к Data-сервису: {e}")
    except Exception as e:
        import traceback; traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Ошибка Gateway при получении сводки: {e}")


@app.get("/api/health", tags=["Утилиты"])
async def health_check_gateway():
    status_ml = {"status": "unreachable"}; status_data = {"status": "unreachable"}
    if httpx_client is None: return {"status": "gateway_uninitialized", "services": {"ml": status_ml, "data": status_data}}
    try:
        ml_resp = await httpx_client.get(f"{cfg.ML_SERVICE_URL}/health"); ml_resp.raise_for_status(); status_ml = ml_resp.json()
    except Exception as e: status_ml["message"] = f"Error: {e}"
    try:
        data_resp = await httpx_client.get(f"{cfg.DATA_SERVICE_URL}/health"); data_resp.raise_for_status(); status_data = data_resp.json()
    except Exception as e: status_data["message"] = f"Error: {e}"
    if status_ml.get("status") == "ok" and status_data.get("status") == "ok":
        return {"status": "ok", "message": "API Gateway и все сервисы готовы.", "services": {"ml": status_ml, "data": status_data}}
    else:
        raise HTTPException(status_code=503, detail="Некоторые сервисы не готовы.", headers={"X-Service-Status": json.dumps({"ml": status_ml, "data": status_data})})