# Cardio Prediction Server

FastAPI service nay tach rieng phan model inference de co the deploy doc lap voi frontend.

## 1. Cai dat

```bash
cd server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Chay local

```bash
cd server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Docs se co tai `http://localhost:8000/docs`.

## 3. Bien moi truong

- `CARDIO_MODEL_PATH`: duong dan toi file model `.joblib`
- `CARDIO_METADATA_PATH`: duong dan toi file metadata `.json`
- `CARDIO_CORS_ORIGINS`: danh sach origin phan tach bang dau phay, mac dinh la `*`

Neu khong set `CARDIO_MODEL_PATH` va `CARDIO_METADATA_PATH`, server se mac dinh doc tu:

- `./models/cardio_xgboost_final.joblib`
- `./models/cardio_xgboost_final_metadata.json`

Luu y: file model `.joblib` phu thuoc vao version `scikit-learn`, `xgboost`, `numpy`, `pandas` va `joblib`. Vi vay `requirements.txt` da duoc pin version de tranh loi runtime kieu `"'str' object has no attribute 'transform'"` khi deploy.

## 4. Deploy len Railway tu GitHub

Neu ban push repo backend nay len GitHub nhu mot repo rieng, Railway se deploy truc tiep tu root cua repo.

Khi tao service moi tren Railway:

1. Chon `Deploy from GitHub repo`.
2. Chon repo backend nay.
3. Railway se doc `railway.json` o root repo de lay start command.
4. Sau khi deploy xong, vao `Networking` va bam `Generate Domain`.

Bien moi truong nen set tren service:

- `CARDIO_CORS_ORIGINS=https://ten-frontend-cua-ban.vercel.app`

Neu frontend cung deploy rieng, set `CARDIO_SERVER_URL` ben frontend thanh domain cua Railway service nay.

Neu sau nay ban dua backend vao chung mot monorepo, khi do moi can set `Root Directory` va tro config toi duong dan tuyet doi cua file `railway.json`.

## 5. API

### `GET /health`

Kiem tra server va model path.

### `POST /predict`

Request body:

```json
{
  "gender": 2,
  "height": 170,
  "weight": 75,
  "ap_hi": 120,
  "ap_lo": 80,
  "cholesterol": 1,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1,
  "age_years": 49.8
}
```

Response sample:

```json
{
  "model": "XGBoost",
  "probability_cardio": 0.241337,
  "prediction": 0,
  "threshold_used": 0.3487778306007385,
  "input": {
    "gender": 2,
    "height": 170,
    "weight": 75,
    "ap_hi": 120,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
    "age_years": 49.8,
    "bmi": 25.95,
    "pulse_pressure": 40,
    "mean_arterial_pressure": 93.33
  }
}
```
