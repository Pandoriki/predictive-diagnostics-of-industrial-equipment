import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

TRAIN_FILE = 'train_FD001.txt' 
TEST_FILE = 'test_FD001.txt'   
RUL_TRUE_FILE = 'RUL_FD001.txt'

USE_XGBOOST_MODEL = False # Явно указываем, что используем CatBoost (не XGBoost)

SEQUENCE_LENGTH = 50   
RUL_CAP = 125          
VALIDATION_SPLIT = 0.2 

OP_SETTING_COLS = ['op_setting_1', 'op_setting_2', 'op_setting_3']
ALL_SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
IRRELEVANT_SENSORS_INDICES = [1, 5, 6, 10, 16, 18, 19] 
NOISY_SENSORS_FOR_FILTER_INDICES = [7, 8, 11, 12, 13, 14, 15, 17, 20, 21]

VIBRATION_SENSORS_FOR_FFT_INDICES = [7, 8, 11, 12] 
FFT_BINS_COUNT = 3 
USE_FFT_FEATURES = True
FFT_SEQUENCE_LENGTH = 10 

USE_DATA_AUGMENTATION = False 
AUGMENTATION_NOISE_FACTOR = 0.01

CATBOOST_CONFIG = {
    'iterations': 500, 
    'learning_rate': 0.05,
    'depth': 8,        
    'loss_function': 'RMSE',
    'eval_metric': 'MAE',   
    'random_seed': 42,
    'verbose': 0,      
    'early_stopping_rounds': 50,
    'l2_leaf_reg': 3,  
    'thread_count': -1
}

N_FOLDS_CV = 5 # K = 5 для K-Fold кросс-валидации

RANDOM_SEED = 42
DEVICE = 'cpu'