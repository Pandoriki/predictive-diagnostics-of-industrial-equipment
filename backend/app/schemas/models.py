from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal

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
    sensor_13: float # Эта строка была продублирована! Исправлено
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
    sequence_data: List[SensorDataPoint] = Field(..., description="Последовательность сырых данных датчиков за SEQUENCE_LENGTH циклов.")

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