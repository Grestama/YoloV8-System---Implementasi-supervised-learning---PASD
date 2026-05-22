## 🦷 YOLOv8 Oral Disease Detection - Deployment Ready Guide

### ⚠️ IMPORTANT (UPDATED ARCHITECTURE)

Project ini menggunakan **2 service terpisah:**

- FastAPI → inference engine
- Streamlit → frontend (via API)

---

## 📁 MODEL SETUP

WAJIB:

```
Project/
└── model/
    └── best.pt
```

Atau set env:

```
MODEL_PATH=/app/model/best.pt
```

---

## 🚀 DOCKER DEPLOYMENT (RECOMMENDED)

### 1. Jalankan

```bash
docker compose up --build
```

---

### 2. Akses

- Streamlit: http://localhost:8501
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

### 3. Arsitektur

```
[Streamlit] ---> [FastAPI] ---> [YOLOv8 Model]
```

---

## ⚙️ ENVIRONMENT VARIABLES

Tambahkan jika perlu:

```
MODEL_PATH=/app/model/best.pt
API_URL=http://api:8000/predict
CONFIDENCE_THRESHOLD=0.5
```

---

## 🧪 API TEST

```bash
curl http://localhost:8000/health
```

---

## 📊 EVALUATION

```bash
python evaluation_report.py
```

Output:

- JSON
- CSV
- HTML

---

## ❗ TROUBLESHOOTING

### Model not found

➡️ Pastikan:

```
model/best.pt
```

---

### API not reachable (Streamlit)

➡️ Cek:

```
docker ps
```

---

### Docker error

Gunakan:

```bash
docker compose up --build
```

❌ BUKAN `docker-compose`

---

## 🔥 STATUS

| Component     | Status   |
| ------------- | -------- |
| API           | ✅       |
| Streamlit     | ✅       |
| Docker        | ✅       |
| Model loading | ✅       |
| Evaluation    | ✅       |
| Deployment    | ✅ READY |

---

## 🎯 FINAL NOTE

Project sudah:

- production-ready
- dockerized
- scalable
- API-decoupled

Deploy via Docker **WAJIB** untuk stability.
