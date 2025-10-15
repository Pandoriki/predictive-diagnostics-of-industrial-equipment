# configs/config.py
import os
import torch

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root: pdm-hackathon-nanobanano
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- Data Filenames ---
TRAIN_FILE = 'train_FD001.txt' 
TEST_FILE = 'test_FD001.txt'   
RUL_TRUE_FILE = 'RUL_FD001.txt'

# --- Preprocessing Parameters ---
SEQUENCE_LENGTH = 50   
RUL_CAP = 125          
VALIDATION_SPLIT = 0.2 

OP_SETTING_COLS = ['op_setting_1', 'op_setting_2', 'op_setting_3']
ALL_SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
IRRELEVANT_SENSORS_INDICES = [1, 5, 6, 10, 16, 18, 19] 
NOISY_SENSORS_FOR_FILTER_INDICES = [7, 8, 11, 12, 13, 14, 15, 17, 20, 21]

# --- НОВОЕ: Настройки для FFT ---
VIBRATION_SENSORS_FOR_FFT_INDICES = [7, 8, 11, 12] 
FFT_BINS_COUNT = 3 
USE_FFT_FEATURES = True

# --- НОВОЕ: Настройки для Аугментации ---
USE_DATA_AUGMENTATION = True
AUGMENTATION_NOISE_FACTOR = 0.01

# --- Model Hyperparameters ---
NUM_EPOCHS = 50        
BATCH_SIZE = 32        
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1
CNN_KERNEL_SIZE = 5    

# Model config для передачи как **kwargs
MODEL_CONFIG = {
    'n_pre': 1,              
    'w_pre': 64,             
    'n_strided': 2,          
    'w_strided': 128,        
    'n_l': 1,                 
    'w_l': 128,              
    'n_dense_post_l': 1,     
    'w_dense_post_l': 64,    
    'cnn_kernel_size': CNN_KERNEL_SIZE,
    'dropout': DROPOUT_RATE,
    'do_pool': True,         
    'stride_amt': 2,         
}

# --- Other Settings ---
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# (Больше нет `class Config` и `cfg = Config()`)