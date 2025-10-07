import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from configs.config import cfg
from src.data_preprocessing import load_data, calculate_rul_for_train, preprocess_features, \
                                    generate_sequences, generate_test_sequences_for_prediction
from src.model_architecture import RULFilterNet
from src.dataset import RULDataset
from src.predict_utils import get_rul_status

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model():
    set_seed(cfg.RANDOM_SEED)
    device = torch.device(cfg.DEVICE)

    print("Загрузка исходных данных...")
    df_train_raw, df_test_raw, df_rul_true = load_data()

    print("Расчет RUL для тренировочного набора...")
    df_train_with_rul = calculate_rul_for_train(df_train_raw, cfg.RUL_CAP)

    print("Предварительная обработка признаков (фильтрация, масштабирование)...")
    df_train_processed, df_test_processed, scaler, selected_features = \
        preprocess_features(df_train_with_rul.copy(), df_test_raw.copy(),
                            fit_scaler=True, 
                            scaler_save_path=os.path.join(cfg.MODELS_DIR, 'scaler.pkl'),
                            features_save_path=os.path.join(cfg.MODELS_DIR, 'selected_features.pkl'))
    
    print(f"Количество выбранных признаков: {len(selected_features)}")

    print("Генерация последовательностей (временных окон) для обучения и тестирования...")
    X_train_sequences, y_train_sequences = generate_sequences(df_train_processed, selected_features, cfg.SEQUENCE_LENGTH)
    
    X_test_for_prediction = generate_test_sequences_for_prediction(df_test_processed, selected_features, cfg.SEQUENCE_LENGTH)
    y_test_true = df_rul_true['RUL_true'].values 
    
    print(f"Форма X_train_sequences: {X_train_sequences.shape}")
    print(f"Форма y_train_sequences: {y_train_sequences.shape}")
    print(f"Форма X_test_for_prediction (для инференса): {X_test_for_prediction.shape}")
    print(f"Форма y_test_true: {y_test_true.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_sequences, y_train_sequences, test_size=cfg.VALIDATION_SPLIT, random_state=cfg.RANDOM_SEED
    )
    
    train_loader, val_loader = RULDataset.create_dataloaders(
        X_train, y_train, X_val, y_val, cfg.BATCH_SIZE
    )
    print(f"Размер тренировочного набора: {len(X_train)}, валидационного: {len(X_val)}")

    print("Инициализация модели RULFilterNet...")
    input_channels = X_train_sequences.shape[2] 
    
    model = RULFilterNet(input_channels=input_channels)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    print("Начало обучения модели...")
    for epoch in range(cfg.NUM_EPOCHS):
        model.train() 
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval() 
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).unsqueeze(1)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                all_val_preds.extend(output.cpu().numpy().flatten())
                all_val_targets.extend(target.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / len(val_loader)
        val_rmse = np.sqrt(mean_squared_error(all_val_targets, all_val_preds))
        val_mae = mean_absolute_error(all_val_targets, all_val_preds)
        
        print(f'Эпоха {epoch+1}/{cfg.NUM_EPOCHS}, Потери тренировки: {avg_train_loss:.4f}, Потери валидации: {avg_val_loss:.4f}, RMSE валидации: {val_rmse:.4f}, MAE валидации: {val_mae:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(cfg.MODELS_DIR, 'rul_prediction_model.pth'))
            print("Лучшая модель сохранена.")

    print("\nОбучение завершено.")

    print("\nОценка финальной модели на тестовых данных...")
    model.load_state_dict(torch.load(os.path.join(cfg.MODELS_DIR, 'rul_prediction_model.pth')))
    model.eval()
    
    X_test_tensor = torch.tensor(X_test_for_prediction, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        test_predictions = model(X_test_tensor).cpu().numpy().flatten()

    final_test_rmse = np.sqrt(mean_squared_error(y_test_true, test_predictions))
    final_test_mae = mean_absolute_error(y_test_true, test_predictions)
    
    print(f"\nИтоговые результаты на тестовых данных:")
    print(f"   RMSE: {final_test_rmse:.4f}")
    print(f"   MAE: {final_test_mae:.4f}")

    print("\nПримеры предсказаний RUL и статусов для нескольких тестовых двигателей:")
    for i in range(min(5, len(test_predictions))): 
        predicted_rul = test_predictions[i]
        true_rul = y_test_true[i]
        status, _, _ = get_rul_status(predicted_rul)
        print(f"   Двигатель №{df_test_raw['unit_number'].unique()[i]} (из test_FD001):")
        print(f"      Прогнозируемый RUL = {predicted_rul:.2f} циклов")
        print(f"      Истинный RUL = {true_rul:.2f} циклов")
        print(f"      Статус = {status}")

if __name__ == "__main__":
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    import torch 
    train_model()