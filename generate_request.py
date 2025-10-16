# generate_request.py
import pandas as pd
import json
import os

# --- Параметры ---
# Путь к папке с сырыми данными (относительно корневой папки проекта)
DATA_RAW_DIR = os.path.join('data', 'raw')
TEST_FILE = 'test_FD001.txt'
# ID агрегата для тестирования
UNIT_ID_TO_TEST = 1
# Длина последовательности (из configs/config.py)
SEQUENCE_LENGTH = 50 

# --- Загрузка и подготовка данных ---
try:
    column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                   [f'sensor_{i}' for i in range(1, 22)]
    
    df_test_raw = pd.read_csv(os.path.join(DATA_RAW_DIR, TEST_FILE), sep=' ', header=None)
    df_test_raw.drop(columns=[26, 27], inplace=True) 
    df_test_raw.columns = column_names

    # Выбираем данные для нужного агрегата
    unit_data = df_test_raw[df_test_raw['unit_number'] == UNIT_ID_TO_TEST]
    
    # Берем последние SEQUENCE_LENGTH записей
    if len(unit_data) < SEQUENCE_LENGTH:
        print(f"Предупреждение: для unit_id={UNIT_ID_TO_TEST} всего {len(unit_data)} записей, что меньше {SEQUENCE_LENGTH}.")
        sequence_data_df = unit_data
    else:
        sequence_data_df = unit_data.tail(SEQUENCE_LENGTH)
    
    # Преобразуем DataFrame в список словарей (формат, совместимый с Pydantic)
    request_data_points = sequence_data_df.to_dict(orient='records')
    
    # Создаем финальный JSON-запрос
    final_json_request = {
        "unit_id": UNIT_ID_TO_TEST,
        "sequence_data": request_data_points
    }

    # Выводим JSON в консоль в красивом виде
    print(json.dumps(final_json_request, indent=2))

except FileNotFoundError:
    print(f"Ошибка: Не удалось найти файл {os.path.join(DATA_RAW_DIR, TEST_FILE)}. Убедитесь, что датасет на месте.")
except Exception as e:
    print(f"Произошла ошибка: {e}")