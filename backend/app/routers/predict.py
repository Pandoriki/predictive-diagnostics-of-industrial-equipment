from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from typing import List
from backend.app.schemas.models import PredictionRequest, PredictionResponse, SensorDataPoint
from backend.app.core.ml_models import MLResources
from backend.app.core.config import settings 
from src.data_preprocessing import preprocess_features 
from src.predict_utils import get_rul_status

router = APIRouter()

# --- Вспомогательная функция для предобработки одной последовательности для инференса ---
def _preprocess_single_sequence_for_inference(raw_sequence_data: List[SensorDataPoint]) -> np.ndarray:
    if MLResources.scaler is None or MLResources.selected_features is None or MLResources.model is None:
        raise ValueError("ML ресурсы не загружены.")

    df_raw_sequence = pd.DataFrame([s.model_dump() for s in raw_sequence_data])

    _current_full_raw_cols = settings._meaningful_raw_columns

    for col in _current_full_raw_cols:
        if col not in df_raw_sequence.columns:
            df_raw_sequence[col] = 0.0
    
    df_for_preprocess_actual = df_raw_sequence[_current_full_raw_cols].copy()

    df_train_dummy = pd.DataFrame(columns=_current_full_raw_cols + ['RUL', 'max_cycle_in_unit'])
    
    _, df_processed_single_sequence, _, _ = \
        preprocess_features(df_train=df_train_dummy.copy(), df_test=df_for_preprocess_actual.copy(), 
                            fit_scaler=False, scaler=MLResources.scaler, is_single_unit_df=True)

    final_flat_features = df_processed_single_sequence[MLResources.selected_features].iloc[-1].values 
    
    if final_flat_features.shape != (len(MLResources.selected_features),):
        raise ValueError(f"Ошибка формы данных после предобработки для инференса CatBoost. Ожидаемо ({len(MLResources.selected_features)},), получено {final_flat_features.shape}. Проверьте `selected_features`.")

    return final_flat_features.astype(np.float32)


@router.post("/predict_rul", response_model=PredictionResponse, tags=["Прогноз RUL"])
async def predict_rul_endpoint(request: PredictionRequest):
    if MLResources.model is None or MLResources.scaler is None or MLResources.selected_features is None:
        raise HTTPException(status_code=503, detail="ML сервис не готов. Модель или скейлер не загружены.")

    try:
        if len(request.sequence_data) != settings.SEQUENCE_LENGTH:
             raise HTTPException(status_code=400, detail=f"Ожидается последовательность данных из {settings.SEQUENCE_LENGTH} циклов. Получено {len(request.sequence_data)}.")

        processed_input_array = _preprocess_single_sequence_for_inference(request.sequence_data)
        
        predicted_rul = MLResources.model.predict(processed_input_array.reshape(1, -1)).item()
        
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