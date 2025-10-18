import pytest
import requests
import json

BASE_URL = "http://chain-impact.ru:8080"
SCHEMA_URL = "/home/ivan/hak/tests/api/schemas"
DATA_URL = "/home/ivan/hak/tests/api/data"

@pytest.fixture
def load_schema():
    def _load(schema_path):
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)
    return _load

