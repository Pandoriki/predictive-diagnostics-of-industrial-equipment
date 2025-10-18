import pytest
import requests
import json
import os
from jsonschema import validate

def load_schema_file(schema_path):
    """Простая функция для загрузки JSON схемы"""
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)

def test_get_history_success_id_1():
    """Тест успешного получения истории оборудования с ID 1"""

    response = requests.get("http://chain-impact.ru:8080/api/history/1")
    

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    

    response_data = response.json()
    

    schema_path = "/home/ivan/hak/tests/api/schemas/equipment_data/get_history_unit_id.json"
    schema = load_schema_file(schema_path)
    
    
    validate(instance=response_data, schema=schema)
    


def test_get_history_success_id_100000():

    response = requests.get("http://chain-impact.ru:8080/api/history/100000")
    

    assert response.status_code >= 400
    
    

   