import os
import torch

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    TRAIN_FILE = 'train_FD001.txt'
    TEST_FILE = 'test_FD001.txt'
    RUL_TRUE_FILE = 'RUL_FD001.txt'

    SEQUENCE_LENGTH = 50
    RUL_CAP = 125
    VALIDATION_SPLIT = 0.2

    OP_SETTING_COLS = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    ALL_SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
    IRRELEVANT_SENSORS_INDICES = [1, 5, 6, 10, 16, 18, 19] 
    NOISY_SENSORS_FOR_FILTER_INDICES = [7, 8, 11, 12, 13, 14, 15, 17, 20, 21]

    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.1
    CNN_KERNEL_SIZE = 5

    MODEL_CONFIG = {
        'n_pre': 1,
        'w_pre': 64,
        'n_strided': 2,
        'w_strided': 128,
        'n_l': 2,
        'w_l': 128,
        'n_dense_post_l': 1,
        'w_dense_post_l': 64,
        'cnn_kernel_size': CNN_KERNEL_SIZE,
        'dropout': DROPOUT_RATE,
        'do_pool': True,
        'stride_pos': "post",
        'stride_amt': 2,
    }
    
    RANDOM_SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = Config()