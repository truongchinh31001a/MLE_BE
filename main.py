import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "artifacts" / "models" / "cardio_xgboost_final.joblib"
DEFAULT_METADATA_PATH = (
    ROOT_DIR / "artifacts" / "metrics" / "cardio_xgboost_final_metadata.json"
)
CARDIO_FIELD_ORDER = [
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "age_years",
    "bmi",
    "pulse_pressure",
    "mean_arterial_pressure",
]


class PayloadValidationError(ValueError):
    def __init__(self, details: list[str]):
        super().__init__("Payload dau vao khong hop le.")
        self.details = details


class PredictionRequest(BaseModel):
    gender: int | str
    height: float | int | str
    weight: float | int | str
    ap_hi: float | int | str
    ap_lo: float | int | str
    cholesterol: int | str
    gluc: int | str
    smoke: int | bool | str
    alco: int | bool | str
    active: int | bool | str
    age_years: float | int | str


class PredictionResponse(BaseModel):
    model: str
    probability_cardio: float
    prediction: int
    threshold_used: float
    input: dict[str, float | int]


def round_to(value: float, digits: int = 2) -> float | None:
    if value is None:
        return None

    return round(value, digits)


def parse_number(value: Any) -> float | int | None:
    if value is None or value == "":
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return value

    try:
        numeric_value = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if numeric_value.is_integer():
        return int(numeric_value)

    return numeric_value


def parse_binary(value: Any) -> int | None:
    if value is True or str(value).strip().lower() == "true":
        return 1

    if value is False or str(value).strip().lower() == "false":
        return 0

    numeric_value = parse_number(value)

    if numeric_value in (0, 1):
        return int(numeric_value)

    return None


def derive_cardio_metrics(values: dict[str, Any]) -> dict[str, float | int | None]:
    height = parse_number(values.get("height"))
    weight = parse_number(values.get("weight"))
    systolic = parse_number(values.get("ap_hi"))
    diastolic = parse_number(values.get("ap_lo"))

    return {
        "bmi": round_to(weight / ((height / 100) ** 2), 2)
        if height and weight
        else None,
        "pulse_pressure": round_to(systolic - diastolic, 0)
        if systolic is not None and diastolic is not None
        else None,
        "mean_arterial_pressure": round_to((systolic + 2 * diastolic) / 3, 2)
        if systolic is not None and diastolic is not None
        else None,
    }


def build_cardio_payload(values: dict[str, Any]) -> dict[str, float | int]:
    normalized = {
        "gender": parse_number(values.get("gender")),
        "height": parse_number(values.get("height")),
        "weight": parse_number(values.get("weight")),
        "ap_hi": parse_number(values.get("ap_hi")),
        "ap_lo": parse_number(values.get("ap_lo")),
        "cholesterol": parse_number(values.get("cholesterol")),
        "gluc": parse_number(values.get("gluc")),
        "smoke": parse_binary(values.get("smoke")),
        "alco": parse_binary(values.get("alco")),
        "active": parse_binary(values.get("active")),
        "age_years": parse_number(values.get("age_years")),
    }
    derived = derive_cardio_metrics(normalized)
    payload = {
        **normalized,
        **derived,
    }
    missing_fields = [
        field for field in CARDIO_FIELD_ORDER if payload.get(field) is None
    ]
    issues: list[str] = []

    if missing_fields:
        issues.append(
            f"Thieu hoac sai dinh dang truong: {', '.join(missing_fields)}"
        )

    if payload["height"] is not None and payload["height"] <= 0:
        issues.append("Chieu cao phai lon hon 0 cm.")

    if payload["weight"] is not None and payload["weight"] <= 0:
        issues.append("Can nang phai lon hon 0 kg.")

    if payload["age_years"] is not None and payload["age_years"] <= 0:
        issues.append("Tuoi phai lon hon 0.")

    if (
        payload["ap_hi"] is not None
        and payload["ap_lo"] is not None
        and payload["ap_hi"] <= payload["ap_lo"]
    ):
        issues.append("Huyet ap tam thu phai lon hon huyet ap tam truong.")

    if payload["gender"] not in (1, 2):
        issues.append("Gioi tinh chi nhan 1 (Nu) hoac 2 (Nam).")

    if payload["cholesterol"] not in (1, 2, 3):
        issues.append("Cholesterol chi nhan cac muc 1, 2, 3.")

    if payload["gluc"] not in (1, 2, 3):
        issues.append("Glucose chi nhan cac muc 1, 2, 3.")

    if issues:
        raise PayloadValidationError(issues)

    return {field: payload[field] for field in CARDIO_FIELD_ORDER}


def get_model_path() -> Path:
    return Path(os.getenv("CARDIO_MODEL_PATH", DEFAULT_MODEL_PATH))


def get_metadata_path() -> Path:
    return Path(os.getenv("CARDIO_METADATA_PATH", DEFAULT_METADATA_PATH))


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Any, dict[str, Any]]:
    model_path = get_model_path()
    metadata_path = get_metadata_path()

    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay model tai {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Khong tim thay metadata tai {metadata_path}")

    model = joblib.load(model_path)
    metadata = metadata_path.read_text(encoding="utf-8")
    return model, json.loads(metadata)


def predict_cardio(payload: dict[str, float | int]) -> dict[str, Any]:
    model, metadata = load_artifacts()
    feature_order = list(getattr(model, "feature_names_in_", CARDIO_FIELD_ORDER))
    frame = pd.DataFrame([{feature: payload[feature] for feature in feature_order}])
    probability_cardio = float(model.predict_proba(frame)[0][1])
    threshold_used = float(metadata["selected_threshold"])

    return {
        "model": metadata.get("final_model_name", "XGBoost"),
        "probability_cardio": round(probability_cardio, 6),
        "prediction": int(probability_cardio >= threshold_used),
        "threshold_used": threshold_used,
        "input": payload,
    }


def parse_cors_origins() -> list[str]:
    raw_value = os.getenv("CARDIO_CORS_ORIGINS", "*").strip()

    if not raw_value:
        return ["*"]

    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


app = FastAPI(
    title="Cardio Prediction Server",
    version="1.0.0",
    description="FastAPI service cho model du doan nguy co tim mach.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(PayloadValidationError)
async def payload_validation_error_handler(_, exc: PayloadValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "message": str(exc),
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_, exc: RequestValidationError):
    details = []

    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", []))
        message = error.get("msg", "Gia tri khong hop le.")
        details.append(f"{location}: {message}" if location else message)

    return JSONResponse(
        status_code=400,
        content={
            "message": "Payload dau vao khong hop le.",
            "details": details,
        },
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_error_handler(_, exc: FileNotFoundError):
    return JSONResponse(
        status_code=500,
        content={
            "message": "Khong the tai model cho prediction server.",
            "error": str(exc),
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "message": "Khong the xu ly du doan luc nay.",
            "error": str(exc),
        },
    )


@app.get("/")
def root():
    return {
        "message": "Cardio prediction server is ready.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    model_path = get_model_path()
    metadata_path = get_metadata_path()
    return {
        "status": "ok",
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "model_exists": model_path.exists(),
        "metadata_exists": metadata_path.exists(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    normalized_payload = build_cardio_payload(payload.model_dump())
    return predict_cardio(normalized_payload)
