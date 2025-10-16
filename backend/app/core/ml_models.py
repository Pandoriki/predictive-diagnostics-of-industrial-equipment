import os
import joblib
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional

from backend.app.core.config import settings # Путь теперь `backend.app.core.config`
from src.data_preprocessing import load_data, preprocess_features # src виден из /app
from src.predict_utils import get_rul_status # src виден из /app


class MLResources:
    model: Optional[CatBoostRegressor] = None
    scaler: Optional[MinMaxScaler] = None
    selected_features: Optional[List[str]] = None
    
    global_historical_data: Optional[pd.DataFrame] = None 
    df_true_test_rul: Optional[pd.DataFrame] = None 
    

def load_ml_resources_on_startup():
    model_path_cbm = os.path.join(settings.MODELS_DIR, 'rul_prediction_model.cbm') 
    scaler_path = os.path.join(settings.MODELS_DIR, 'scaler.pkl')
    features_path = os.path.join(settings.MODELS_DIR, 'selected_features.pkl')

    if not os.path.exists(model_path_cbm) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
        raise RuntimeError(
            f"Не найдены артефакты ML для старта сервиса. Убедитесь, что 'train_model.py' был запущен. "
            f"Ожидаемые пути: {model_path_cbm}, {scaler_path}, {features_path}"
        )

    try:
        MLResources.scaler = joblib.load(scaler_path)
        print(f"[ML_RESOURCES] Scaler загружен из {scaler_path}")

        MLResources.selected_features = joblib.load(features_path)
        print(f"[ML_RESOURCES] Список признаков загружен из {features_path} ({len(MLResources.selected_features)} признаков).")

        MLResources.model = CatBoostRegressor()
        MLResources.model.load_model(model_path_cbm)
        print(f"[ML_RESOURCES] CatBoost модель загружена из {model_path_cbm}.")

        _, global_historical_data_raw, df_true_test_rul_temp = load_data(settings.DATA_RAW_DIR)
        
        MLResources.global_historical_data = global_historical_data_raw.copy()
        MLResources.df_true_test_rul = df_true_test_rul_temp
        
        max_time_per_unit = MLResources.global_historical_data.groupby('unit_number')['time_in_cycles'].max()
        MLResources.global_historical_data = MLResources.global_historical_data.merge(
            max_time_per_unit.rename('max_cycle_in_unit'), on='unit_number', how='left')

        print(f"[ML_RESOURCES] Имитация исторической базы данных загружена: {MLResources.global_historical_data.shape}")
        
    except Exception as e:
        import traceback
        print(f"[ML_RESOURCES] КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить ML ресурсы: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Не удалось инициализировать ML ресурсы. Сервис не может запуститься: {e}")