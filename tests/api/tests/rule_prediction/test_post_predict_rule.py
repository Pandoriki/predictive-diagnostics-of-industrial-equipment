import pytest
import requests
import json
import os
from jsonschema import validate
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

def load_schema_file(schema_path):
    """Простая функция для загрузки JSON схемы"""
    logger.info(f"Загрузка схемы из файла: {schema_path}")
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    logger.info("Схема успешно загружена")
    return schema

def test_post_predict_rul_success():
    """Тест успешного прогнозирования RUL"""
    logger.info("=== НАЧАЛО ТЕСТА: Успешное прогнозирование RUL ===")
    
    base_dir = "/home/ivan/hak/tests/api"
    logger.info(f"Базовая директория: {base_dir}")

    # Загрузка тестовых данных
    data_path = os.path.join(base_dir, "data/rule_prediction/post_predict_rule.json")
    logger.info(f"Загрузка тестовых данных из: {data_path}")
    
    with open(data_path, encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info("Тестовые данные успешно загружены")
    
    # Отправка POST запроса
    logger.info("Отправка POST запроса на http://chain-impact.ru:8080/api/predict_rul")
    response = requests.post("http://chain-impact.ru:8080/api/predict_rul", json=test_data)
    logger.info(f"Получен ответ с статус кодом: {response.status_code}")
    
    # Проверка статус кода
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    logger.info("✓ Статус код 200 - УСПЕШНО")
    
    # Парсинг JSON ответа
    response_data = response.json()
    logger.info("Ответ успешно преобразован в JSON")
    logger.info(f"Содержимое ответа: {json.dumps(response_data, indent=2, ensure_ascii=False)}")

    # Загрузка и валидация схемы
    schema_path = os.path.join(base_dir, "schemas/rule prediction/post_predict_rule.json")
    logger.info(f"Загрузка JSON схемы из: {schema_path}")
    schema = load_schema_file(schema_path)

    logger.info("Начало валидации ответа по схеме...")
    validate(instance=response_data, schema=schema)
    logger.info("✓ Валидация по схеме - УСПЕШНО")

    # Дополнительные проверки
    logger.info("Выполнение дополнительных проверок...")
    assert "predicted_rul" in response_data, "Response should contain 'predicted_rul' field"
    logger.info("✓ Поле 'predicted_rul' присутствует в ответе")
    
    assert isinstance(response_data["predicted_rul"], (int, float)), "predicted_rul should be a number"
    logger.info(f"✓ Поле 'predicted_rul' содержит числовое значение: {response_data['predicted_rul']}")
    
    logger.info("=== ТЕСТ ЗАВЕРШЕН УСПЕШНО ===")

def test_post_predict_rul_invalid_data():
    """Тест обработки невалидных данных"""
    logger.info("=== НАЧАЛО ТЕСТА: Обработка невалидных данных ===")
    
    base_dir = "/home/ivan/hak/tests/api"
    logger.info(f"Базовая директория: {base_dir}")
    
    # Проверка существования файла с невалидными данными
    data_path = os.path.join(base_dir, "data/rule_prediction/invalid_predict_rule.json")
    logger.info(f"Проверка существования файла: {data_path}")
    
    if not os.path.exists(data_path):
        logger.warning(f"Файл {data_path} не найден, пропуск теста")
        pytest.skip(f"File {data_path} not found, skipping test")
    
    # Загрузка невалидных данных
    logger.info("Загрузка невалидных тестовых данных")
    with open(data_path, encoding="utf-8") as f:
        invalid_data = json.load(f)
    logger.info(f"Невалидные данные загружены: {json.dumps(invalid_data, indent=2, ensure_ascii=False)}")
    
    # Отправка POST запроса с невалидными данными
    logger.info("Отправка POST запроса с невалидными данными...")
    response = requests.post("http://chain-impact.ru:8080/api/predict_rul", json=invalid_data)
    logger.info(f"Получен ответ с статус кодом: {response.status_code}")
    
    # Проверка, что сервер вернул ошибку
    assert response.status_code >= 400, f"Expected error status (4xx), got {response.status_code}"
    logger.info(f"✓ Сервер вернул ожидаемый код ошибки: {response.status_code}")
    
    # Логирование содержимого ответа об ошибке (если есть)
    if response.text:
        logger.info(f"Содержимое ответа об ошибке: {response.text}")
    
    logger.info("=== ТЕСТ ЗАВЕРШЕН УСПЕШНО ===")