FROM python:3.10.16-slim

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
