import numpy as np
import random
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor, Pool

import configs.config as cfg
from src.data_preprocessing import load_data, calculate_rul_for_train, preprocess_features, \
                                    generate_flat_features_for_boosting, \
                                    generate_flat_test_features_for_boosting
from src.predict_utils import get_rul_status, mean_absolute_percentage_error


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def train_model():
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    set_seed(cfg.RANDOM_SEED)

    print("Загрузка исходных данных...")
    df_train_raw, df_test_raw, df_rul_true = load_data()

    print("Расчет RUL для тренировочного набора...")
    df_train_with_rul = calculate_rul_for_train(df_train_raw, cfg.RUL_CAP)

    print("Предварительная обработка признаков (фильтрация, масштабирование, FFT)...")
    df_train_processed, df_test_processed, scaler, selected_features = \
        preprocess_features(df_train_with_rul.copy(), df_test_raw.copy(),
                            fit_scaler=True, 
                            scaler_save_path=os.path.join(cfg.MODELS_DIR, 'scaler.pkl'),
                            features_save_path=os.path.join(cfg.MODELS_DIR, 'selected_features.pkl'))
    
    print(f"Количество выбранных признаков: {len(selected_features)}")

    print("\n--- ИСПОЛЬЗУЕМ CATBOOST МОДЕЛЬ С K-FOLD КРОСС-ВАЛИДАЦИЕЙ (K=5) ---")
    print("Генерация плоских признаков для обучения CatBoost...")
    
    X_full_train_for_folds, y_full_train_for_folds = generate_flat_features_for_boosting(
        df_train_processed, selected_features, cfg.SEQUENCE_LENGTH)
    
    X_final_test_predict_ready = generate_flat_test_features_for_boosting(
        df_test_processed, selected_features, cfg.SEQUENCE_LENGTH)
    y_final_test_true = df_rul_true['RUL_true'].values 


    print(f"Форма X_full_train_for_folds: {X_full_train_for_folds.shape}")
    print(f"Форма y_full_train_for_folds: {y_full_train_for_folds.shape}")
    print(f"Форма X_final_test_predict_ready: {X_final_test_predict_ready.shape}")
    print(f"Форма y_final_test_true: {y_final_test_true.shape}")

    N_SPLITS = cfg.N_FOLDS_CV
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=cfg.RANDOM_SEED)

    oof_preds_for_meta = np.zeros(len(X_full_train_for_folds))
    final_test_predictions_per_fold = []

    fold_metrics = {
        'rmse': [],
        'mae': [],
        'mape': []
    }
    
    print(f"\n--- Начало обучения с {N_SPLITS}-фолдовой кросс-валидацией ---")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full_train_for_folds)):
        print(f"\n--- Обучение Фолд {fold + 1}/{N_SPLITS} ---")

        X_train, X_val = X_full_train_for_folds[train_idx], X_full_train_for_folds[val_idx]
        y_train, y_val = y_full_train_for_folds[train_idx], y_full_train_for_folds[val_idx]

        cat_model = CatBoostRegressor(**cfg.CATBOOST_CONFIG)
        cat_model.fit(X_train, y_train,
                      eval_set=Pool(X_val, y_val),
                      early_stopping_rounds=cfg.CATBOOST_CONFIG['early_stopping_rounds'],
                      verbose=0) 
        
        val_preds = cat_model.predict(X_val)
        oof_preds_for_meta[val_idx] = val_preds

        test_preds_this_fold = cat_model.predict(X_final_test_predict_ready)
        final_test_predictions_per_fold.append(test_preds_this_fold)

        rmse_val = np.sqrt(mean_squared_error(y_val, val_preds))
        mae_val = mean_absolute_error(y_val, val_preds)
        mape_val = mean_absolute_percentage_error(y_val, val_preds)

        fold_metrics['rmse'].append(rmse_val)
        fold_metrics['mae'].append(mae_val)
        fold_metrics['mape'].append(mape_val)
        
        print(f"   Фолд {fold + 1} завершен. RMSE на валидации: {rmse_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_val:.2f}%")

    print("\n--- Обучение с кросс-валидацией завершено ---")
    
    avg_rmse_cv = np.mean(fold_metrics['rmse'])
    avg_mae_cv = np.mean(fold_metrics['mae'])
    avg_mape_cv = np.mean(fold_metrics['mape'])

    print(f"\nСредние метрики CatBoost на кросс-валидации (обучающие данные):")
    print(f"   RMSE (CV): {avg_rmse_cv:.4f}")
    print(f"   MAE (CV): {avg_mae_cv:.4f}")
    print(f"   MAPE (CV): {avg_mape_cv:.2f}%")

    print("\nОбучение финальной модели CatBoost на всех тренировочных данных...")
    final_cat_model = CatBoostRegressor(**cfg.CATBOOST_CONFIG)
    final_cat_model.fit(X_full_train_for_folds, y_full_train_for_folds, verbose=0)
    final_cat_model.save_model(os.path.join(cfg.MODELS_DIR, 'rul_prediction_model.cbm')) 
    print("--- Финальная CatBoost модель (обученная на всех данных) сохранена! ---")

    avg_final_test_predictions = np.mean(final_test_predictions_per_fold, axis=0)
    
    final_test_rmse = np.sqrt(mean_squared_error(y_final_test_true, avg_final_test_predictions))
    final_test_mae = np.absolute(y_final_test_true - avg_final_test_predictions).mean() 
    final_test_mape = mean_absolute_percentage_error(y_final_test_true, avg_final_test_predictions)

    print(f"\n--- Итоговые результаты на ОТЛОЖЕННОМ тестовом наборе (CatBoost, усреднённые по фолдам): ---")
    print(f"   RMSE: {final_test_rmse:.4f}")
    print(f"   MAE: {final_test_mae:.4f}")
    print(f"   MAPE: {final_test_mape:.2f}%")

    plt.figure(figsize=(10, 6))
    sns.histplot(y_final_test_true - avg_final_test_predictions, bins=50, kde=True, color='blue')
    plt.title('Гистограмма ошибок предсказания RUL (CatBoost, усреднённые)')
    plt.xlabel('Ошибка предсказания (Истинный RUL - Прогнозируемый RUL)')
    plt.ylabel('Частота')
    plt.savefig(os.path.join(cfg.MODELS_DIR, 'prediction_errors_histogram_catboost_cv.png'))
    plt.show()

    print("\nПримеры предсказаний RUL и статусов для нескольких тестовых двигателей (CatBoost):")
    for i in range(min(5, len(avg_final_test_predictions))): 
        predicted_rul = avg_final_test_predictions[i]
        true_rul = y_final_test_true[i]
        status, _, _ = get_rul_status(predicted_rul)
        print(f"   Двигатель №{df_test_raw['unit_number'].unique()[i]}:")
        print(f"      Прогнозируемый RUL = {predicted_rul:.2f} циклов")
        print(f"      Истинный RUL = {true_rul:.2f} циклов")
        print(f"      Статус = {status}")

if __name__ == "__main__":
    import catboost
    import matplotlib
    import seaborn
    train_model()