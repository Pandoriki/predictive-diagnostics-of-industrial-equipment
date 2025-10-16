# backend/app/core/config.py
import configs.config as cfg_module # Прямой импорт, так как `configs` находится в /app/configs/

class Settings:
    def __init__(self):
        self.DATA_RAW_DIR = cfg_module.DATA_RAW_DIR
        self.MODELS_DIR = cfg_module.MODELS_DIR
        self.TRAIN_FILE = cfg_module.TRAIN_FILE
        self.TEST_FILE = cfg_module.TEST_FILE
        self.RUL_TRUE_FILE = cfg_module.RUL_TRUE_FILE
        self.SEQUENCE_LENGTH = cfg_module.SEQUENCE_LENGTH
        self.RUL_CAP = cfg_module.RUL_CAP
        self.OP_SETTING_COLS = cfg_module.OP_SETTING_COLS
        self.ALL_SENSOR_COLS = cfg_module.ALL_SENSOR_COLS
        self.IRRELEVANT_SENSORS_INDICES = cfg_module.IRRELEVANT_SENSORS_INDICES
        self.NOISY_SENSORS_FOR_FILTER_INDICES = cfg_module.NOISY_SENSORS_FOR_FILTER_INDICES
        self.VIBRATION_SENSORS_FOR_FFT_INDICES = cfg_module.VIBRATION_SENSORS_FOR_FFT_INDICES
        self.FFT_BINS_COUNT = cfg_module.FFT_BINS_COUNT
        self.USE_FFT_FEATURES = cfg_module.USE_FFT_FEATURES
        self.CATBOOST_CONFIG = cfg_module.CATBOOST_CONFIG
        self.RANDOM_SEED = cfg_module.RANDOM_SEED
        self.DEVICE = cfg_module.DEVICE

        self._raw_data_full_column_names_from_txt = cfg_module._raw_data_full_column_names_from_txt
        self._meaningful_raw_columns = cfg_module._meaningful_raw_columns
        
settings = Settings()