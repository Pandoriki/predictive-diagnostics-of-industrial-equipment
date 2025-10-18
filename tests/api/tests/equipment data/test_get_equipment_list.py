import pytest
import requests
import json
import os
from jsonschema import validate

def load_schema_file(schema_path):
   
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)

def test_get_equipment_list_success():

    response = requests.get("http://chain-impact.ru:8080/api/equipment_list")
    

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
 
    response_data = response.json()

    schema_path = "/home/ivan/hak/tests/api/schemas/equipment_data/get_equipment_list.json"
    schema = load_schema_file(schema_path)
    

    validate(instance=response_data, schema=schema)

def test_get_equipment_list_invalid_method():


    response = requests.post("http://chain-impact.ru:8080/api/equipment_list")
  
    assert response.status_code >= 400, f"Expected 405 for POST request, got {response.status_code}"