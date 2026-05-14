# Health Engine — Deployment Guide

## Overview

Health Engine is a production-grade ML system for recovery prediction from wearable health data. This guide covers deployment in multiple environments.

---

## Local Development Setup

### Prerequisites

- Python 3.10+
- pip or conda
- Git

### Installation Steps

1. **Clone and navigate to project:**
```bash
cd /d/health-engine
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Run full pipeline:**
```bash
python scripts/run_full_pipeline.py
```

This will:
- Load raw data
- Clean and preprocess
- Calculate recovery scores
- Train the model
- Generate explainability reports

### Running Services Locally

**Option 1: API Server**
```bash
python -m uvicorn app.main:app --reload
```
- API available at: http://localhost:8000
- Swagger docs at: http://localhost:8000/docs

**Option 2: Gradio UI**
```bash
python app.py
```
- Gradio UI available at: http://localhost:7860

**Option 3: Both (different terminals)**
```bash
# Terminal 1
python -m uvicorn app.main:app --reload

# Terminal 2
python app.py
```

---

## Docker Deployment

### Build & Run Single Container

**Build image:**
```bash
docker build -t health-engine:latest .
```

**Run API:**
```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --name health-engine-api \
  health-engine:latest
```

**Run Gradio UI:**
```bash
docker run -d \
  -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --name health-engine-gradio \
  health-engine:latest \
  python app.py
```

### Docker Compose (Recommended)

**Start both services:**
```bash
docker-compose up -d
```

**Check status:**
```bash
docker-compose ps
```

**View logs:**
```bash
docker-compose logs -f api
docker-compose logs -f gradio
```

**Stop services:**
```bash
docker-compose down
```

---

## Hugging Face Spaces Deployment

### Prerequisites

- Hugging Face account
- Git with LFS (for large model files)
- Hugging Face CLI: `pip install huggingface-hub`

### Setup Steps

1. **Create new Space:**
   - Go to https://huggingface.co/new-space
   - Choose "Gradio" as the Space type
   - Name it: `health-engine`

2. **Initialize repo:**
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/health-engine
cd health-engine
```

3. **Copy files:**
```bash
# Copy project files
cp -r /d/health-engine/* .

# Or copy specific files:
cp app.py .
cp requirements.txt .
cp -r src/ models/ .
```

4. **Update requirements.txt:**
```bash
# Make sure all dependencies are listed
cat requirements.txt
```

5. **Create app.py for entry point:**
```bash
cp /d/health-engine/app.py .
```

6. **Add secrets (if needed):**
   - Go to Space Settings → Repository secrets
   - Add any sensitive credentials (API keys, etc.)

7. **Push to Hugging Face:**
```bash
git add .
git commit -m "Initial Health Engine deployment"
git push
```

The Space will automatically build and deploy!

**Access your Space:**
- https://huggingface.co/spaces/YOUR_USERNAME/health-engine

### Alternative: Deploy API as Docker Container

1. Create Space with Docker runtime
2. Follow Docker build steps above
3. Push Dockerfile instead of the Gradio app

---

## API Usage Examples

### Single Prediction

**Using curl:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hrv": 60,
    "resting_heart_rate": 62,
    "day_strain": 5.2,
    "spo2": 97,
    "sleep_performance_percentage": 87,
    "average_hr_bpm": 75
  }'
```

**Using Python:**
```python
import requests

url = "http://localhost:8000/api/v1/predict"
data = {
    "hrv": 60,
    "resting_heart_rate": 62,
    "day_strain": 5.2,
    "spo2": 97,
    "sleep_performance_percentage": 87,
    "average_hr_bpm": 75
}

response = requests.post(url, json=data)
print(response.json())
```

### Batch Prediction with CSV

**Upload file:**
```bash
curl -X POST http://localhost:8000/api/v1/predict-csv \
  -F "file=@predictions.csv"
```

### Batch Prediction with JSON

```bash
curl -X POST http://localhost:8000/api/v1/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "hrv": 60,
        "resting_heart_rate": 62,
        "day_strain": 5.2,
        "spo2": 97,
        "sleep_performance_percentage": 87,
        "average_hr_bpm": 75
      },
      {
        "hrv": 55,
        "resting_heart_rate": 65,
        "day_strain": 7.5,
        "spo2": 96.5,
        "sleep_performance_percentage": 82,
        "average_hr_bpm": 78
      }
    ]
  }'
```

---

## CLI Tools

### Single Prediction

```bash
python scripts/predict_single.py \
  --hrv 60 \
  --rhr 62 \
  --strain 5.2 \
  --spo2 97 \
  --sleep-perf 87 \
  --avg-hr 75
```

Output as JSON:
```bash
python scripts/predict_single.py \
  --hrv 60 \
  --rhr 62 \
  --strain 5.2 \
  --json
```

### Batch Prediction

```bash
python scripts/predict_batch.py input.csv -o output.csv
```

---

## Environment Variables

Key environment variables for configuration:

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
ENV=production

# Model Path
MODEL_PATH=models/trained/recovery_random_forest.pkl

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/health_engine.log

# Data
DATA_RAW_PATH=data/raw
DATA_PROCESSED_PATH=data/processed
```

See `.env.example` for all options.

---

## Monitoring & Health Checks

### Health Endpoint

```bash
curl http://localhost:8000/health
```

### Kubernetes Health Check

Configure readiness probe:
```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

---

## Scaling Considerations

1. **Model Loading:** Model is loaded once on startup and cached
2. **Concurrency:** FastAPI/Uvicorn handles concurrent requests
3. **Batch Processing:** Supports up to 1000 records per batch
4. **Memory:** ~500MB RAM required for model + dependencies

For production scaling:
- Run multiple API instances behind load balancer
- Use Gunicorn/Uvicorn workers
- Deploy on Kubernetes or cloud platform

---

## Troubleshooting

### Model Not Loading

```bash
# Check model path
ls -la models/trained/

# Verify path in config
grep MODEL_PATH config/config.py .env
```

### API Won't Start

```bash
# Check port availability
lsof -i :8000

# Run with verbose logging
python -m uvicorn app.main:app --log-level debug
```

### Low Prediction Accuracy

1. Check input data ranges
2. Verify feature engineering is working
3. Compare with training data statistics
4. Retrain model with new data

---

## Performance Benchmarks

- **Single Prediction:** ~50ms
- **Batch (100 records):** ~500ms
- **CSV Upload & Process:** ~1s per 100 records
- **API Response:** <100ms (99th percentile)

---

## Security Best Practices

1. **API Authentication** (recommended for production):
   - Add API key validation
   - Implement OAuth2
   - Use HTTPS/TLS

2. **Data Privacy:**
   - Do not log sensitive health data
   - Validate input ranges
   - Sanitize file uploads

3. **Model Protection:**
   - Keep model path private
   - Version and backup models
   - Monitor for model drift

---

## Support & Documentation

- **API Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health
- **GitHub:** [Repository URL]

---

## Version History

- **v1.0.0** (May 2026)
  - Initial production release
  - FastAPI with batch prediction
  - Gradio UI
  - Docker deployment support
  - Hugging Face Spaces integration

---

Last Updated: May 2026
