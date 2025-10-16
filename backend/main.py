import os
import numpy as np
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request # Добавлен Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager 
from pydantic import BaseModel, Field # Pydantic BaseModel для RequestBody
from typing import List, Optional, Literal # Используем Literal для строгой типизации
import httpx # Для HTTP-запросов к другим микросервисам (Data-Service, ML-Service)

# Импорты конфигураций
import configs.config as cfg # Для SEQUENCE_LENGTH
# Другие ML-зависимости не нужны, т.к. этот сервис не занимается ML/данными

# --- Pydantic модели (изменен импорт, они должны быть общими для бэкенда) ---
# Предполагаем, что Pydantic-модели будут импортироваться из общего файла в `schemas`
# Или скопировать сюда. Для монолита это был бы тот же main.py
# В нашей текущей модульной архитектуре `backend/main.py` (НЕ `backend/app/main.py` из предыдущей структуры)
# он будет напрямую обращаться к MLResources.

# Новая схема
# Оставим тут. Все Pydantic модели (они могут быть в отдельном файле /schemas/models.py,
# но пока тут для упрощения.)

_full_raw_column_names_for_datapoint_parsing = ['unit_number', 'time_in_cycles'] + cfg.OP_SETTING_COLS + cfg.ALL_SENSOR_COLS

class SensorDataPoint(BaseModel):
    __pydantic_config__ = {'extra': 'ignore'} # Игнорировать поля, которых нет в модели

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

# --- GLOBAL HTTPX CLIENT (для запросов к микросервисам) ---
# Инициализируем один раз для эффективного использования.
httpx_client: Optional[httpx.AsyncClient] = None

# --- Контекстный менеджер FastAPI для событий Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("API Gateway: Запуск сервиса...")
    global httpx_client
    httpx_client = httpx.AsyncClient() # Инициализация клиента HTTP
    yield 
    print("API Gateway: Завершение работы сервиса. Закрытие HTTP клиента...")
    await httpx_client.aclose()


app = FastAPI(
    title="Предиктивная Диагностика: API Gateway",
    description="Основной API-шлюз, маршрутизирующий запросы к ML и Data-микросервисам.",
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


# --- API ENDPOINTS (Переадресация запросов к другим микросервисам) ---

@app.post("/api/predict_rul", response_model=PredictionResponse, tags=["Прогноз RUL"]) 
async def predict_rul_endpoint(request: PredictionRequest):
    if httpx_client is None:
        raise HTTPException(status_code=500, detail="HTTP клиент не инициализирован.")
    
    # Перенаправляем запрос к ML-Service
    try:
        ml_response = await httpx_client.post(f"{cfg.ML_SERVICE_URL}/predict_rul", json=request.model_dump())
        ml_response.raise_for_status() # Выбросить исключение для кодов 4xx/5xx
        return PredictionResponse(**ml_response.json())
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"ML Service Error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Невозможно подключиться к ML-сервису: {e}")
    except Exception as e:
        import traceback
        print(f"[{datetime.now().isoformat()}] Prediction Error in Gateway: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка Gateway при перенаправлении прогноза: {e}")


@app.get("/api/history/{unit_id}", response_model=HistoryResponse, tags=["Данные Оборудования"]) 
async def get_unit_history(unit_id: int):
    if httpx_client is None:
        raise HTTPException(status_code=500, detail="HTTP клиент не инициализирован.")

    try:
        data_response = await httpx_client.get(f"{cfg.DATA_SERVICE_URL}/history/{unit_id}")
        data_response.raise_for_status()
        return HistoryResponse(**data_response.json())
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Data Service Error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Невозможно подключиться к Data-сервису: {e}")
    except Exception as e:
        import traceback
        print(f"[{datetime.now().isoformat()}] History Error in Gateway: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка Gateway при перенаправлении истории: {e}")


@app.get("/api/equipment_list", response_model=List[int], tags=["Данные Оборудования"]) 
async def get_equipment_list():
    if httpx_client is None:
        raise HTTPException(status_code=500, detail="HTTP клиент не инициализирован.")
    
    try:
        data_response = await httpx_client.get(f"{cfg.DATA_SERVICE_URL}/equipment_list")
        data_response.raise_for_status()
        return data_response.json() # Возвращаем список ints напрямую
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Data Service Error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Невозможно подключиться к Data-сервису: {e}")
    except Exception as e:
        import traceback
        print(f"[{datetime.now().isoformat()}] Equipment List Error in Gateway: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка Gateway при получении списка: {e}")


@app.get("/api/status_summary", response_model=List[EquipmentStatusSummary], tags=["Обзор Статусов"])
async def get_all_equipment_status_summary():
    if httpx_client is None:
        raise HTTPException(status_code=500, detail="HTTP клиент не инициализирован.")

    try:
        data_response = await httpx_client.get(f"{cfg.DATA_SERVICE_URL}/status_summary")
        data_response.raise_for_status()
        return data_response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Data Service Error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Невозможно подключиться к Data-сервису: {e}")
    except Exception as e:
        import traceback
        print(f"[{datetime.now().isoformat()}] Status Summary Error in Gateway: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка Gateway при получении сводки: {e}")


@app.get("/api/health", tags=["Утилиты"])
async def health_check_gateway():
    """Проверяет работоспособность Gateway и базовых подключений к микросервисам."""
    
    status_ml = {"status": "unreachable"}
    status_data = {"status": "unreachable"}
    
    if httpx_client is None:
        return {"status": "gateway_uninitialized", "message": "HTTP client is not ready. Gateway cannot function.", "services": {"ml": status_ml, "data": status_data}}

    try:
        ml_resp = await httpx_client.get(f"{cfg.ML_SERVICE_URL}/health")
        ml_resp.raise_for_status()
        status_ml = ml_resp.json()
    except Exception as e:
        status_ml["message"] = f"Error: {e}"

    try:
        data_resp = await httpx_client.get(f"{cfg.DATA_SERVICE_URL}/health")
        data_resp.raise_for_status()
        status_data = data_resp.json()
    except Exception as e:
        status_data["message"] = f"Error: {e}"

    if status_ml.get("status") == "ok" and status_data.get("status") == "ok":
        return {"status": "ok", "message": "API Gateway и все сервисы готовы.", "services": {"ml": status_ml, "data": status_data}}
    else:
        raise HTTPException(status_code=503, detail="Некоторые сервисы не готовы или недоступны.", 
                            headers={"X-Service-Status": json.dumps({"ml": status_ml, "data": status_data})})