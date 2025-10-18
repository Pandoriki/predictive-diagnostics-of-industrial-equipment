import pytest
import requests
import json
import os
from jsonschema import validate

def load_schema_file(schema_path):
    
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)

def test_get_health_success():
    
    
    response = requests.get("http://chain-impact.ru:8080/api/health")
    
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    
   
    response_text = response.text
    assert isinstance(response_text, str), "Response should be a string"
    assert len(response_text) > 0, "Response should not be empty"

def test_get_health_invalid_method():
    
 
    response = requests.post("http://chain-impact.ru:8080/api/health")
    
    
    assert response.status_code >= 400, f"Expected 405 for POST request, got {response.status_code}"