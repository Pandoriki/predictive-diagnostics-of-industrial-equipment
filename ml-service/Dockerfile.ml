FROM python:3.9-slim-buster
WORKDIR /app

COPY ml-service/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements.txt
RUN rm ./requirements.txt

# --- Копируем конфиг-файл ---
COPY configs/ /app/configs/
# --- Копируем ML-логику из src/ ---
COPY src/ /app/src/
# --- Копируем ML-артефакты ---
COPY models/ /app/models/

# --- Копируем само приложение ML-сервиса ---
COPY ml-service/main.py /app/main.py

ENV PYTHONPATH=/app:$PYTHONPATH 
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]