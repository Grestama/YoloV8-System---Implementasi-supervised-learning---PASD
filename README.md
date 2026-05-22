# 🦷 YOLOv8 Oral Disease Detection (Production Ready)

## 🎯 Ringkasan

Aplikasi ini terdiri dari:

- FastAPI → backend inference YOLOv8
- Streamlit → frontend (via API)
- Docker → deployment utama

---

## ⚠️ PENTING

Frontend **tidak bisa jalan sendiri**
➡️ Harus terhubung ke API

---

## 🚀 CARA MENJALANKAN (RECOMMENDED)

### 1. Jalankan Docker

```bash
docker compose up --build
```

---

### 2. Akses Aplikasi

- Frontend: http://localhost:8501
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

## 📁 MODEL

Pastikan model tersedia:

```
model/best.pt
```

Atau gunakan environment variable:

```bash
MODEL_PATH=/app/model/best.pt
```

---

## 🧪 MODE MANUAL (DEV ONLY)

### Jalankan API

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### Jalankan Streamlit

```bash
streamlit run streamlit_app.py
```

---

## 🔍 FITUR

- Upload gambar → deteksi penyakit
- Bounding box + confidence
- Inference time & FPS
- Summary hasil deteksi
- Save hasil gambar

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

## 📦 FILE UTAMA

- `api.py` → backend inference
- `streamlit_app.py` → frontend
- `utils.py` → helper
- `config.py` → konfigurasi
- `evaluation_report.py` → evaluasi model
- `export_onnx.py` → export model

---

## ❗ TROUBLESHOOTING

### API tidak connect

Pastikan:

```bash
docker ps
```

---

### Model tidak ditemukan

Pastikan:

```
model/best.pt
```

---

### Docker error

Gunakan:

```bash
docker compose up --build
```

❌ bukan `docker-compose`

---

## 🔥 STATUS

| Component  | Status |
| ---------- | ------ |
| Model      | ✅     |
| API        | ✅     |
| Frontend   | ✅     |
| Docker     | ✅     |
| Evaluation | ✅     |

---

## 🎯 FINAL

Project siap:

- deploy
- demo
- production (basic)

Gunakan Docker untuk hasil paling stabil.
