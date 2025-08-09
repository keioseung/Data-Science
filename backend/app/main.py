from __future__ import annotations

import io
import json
import uuid
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


class PreprocessOptions(BaseModel):
    impute_missing: bool = True
    remove_outliers: bool = True
    normalize: bool = False
    standardize: bool = False
    encode_categorical: bool = True


class TargetRequest(BaseModel):
    target_variable: str


class TrainRequest(BaseModel):
    model_name: str = "Random Forest"
    test_size: float = 0.2
    random_state: int = 42


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class SessionState(BaseModel):
    session_id: str
    columns: list[str]
    problem_type: Optional[str] = None  # classification | regression
    target_variable: Optional[str] = None
    # Internal objects - not JSON serializable in default, keep out of responses
    # Stored separately in SESSION_STORE


app = FastAPI(title="DataFlow API", version="1.0.0")


def get_cors_origins_from_env() -> list[str]:
    import os

    raw = os.getenv("BACKEND_CORS_ORIGINS", "*").strip()
    if raw == "*" or raw == "":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins_from_env(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Very simple in-memory store. For production, replace with DB or persistent volume
SESSION_STORE: Dict[str, Dict[str, Any]] = {}


def _read_file_to_dataframe(upload: UploadFile) -> pd.DataFrame:
    name = upload.filename or "uploaded"
    content = upload.file.read()
    buf = io.BytesIO(content)
    lower = name.lower()

    if lower.endswith(".csv") or upload.content_type in {"text/csv", "application/csv"}:
        df = pd.read_csv(buf)
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        df = pd.read_excel(buf)
    elif lower.endswith(".json") or upload.content_type == "application/json":
        # try json lines or records
        try:
            df = pd.read_json(buf, orient="records")
        except Exception:
            buf.seek(0)
            df = pd.read_json(buf, lines=True)
    elif lower.endswith(".txt"):
        df = pd.read_csv(buf, sep="\t")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, Excel, JSON, or TXT.")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file contains no rows.")
    return df


def _basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    total_rows = int(df.shape[0])
    total_cols = int(df.shape[1])
    missing_ratio = float(df.isna().sum().sum() / (total_rows * total_cols)) if total_rows * total_cols > 0 else 0.0
    # Simple outlier heuristic: z-score > 3 for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    outliers = 0
    if not numeric_df.empty:
        z = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)
        outliers = int((np.abs(z) > 3).sum().sum())
    preview = df.head(5).to_dict(orient="records")
    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_ratio": round(missing_ratio * 100, 2),
        "outliers": outliers,
        "preview": preview,
        "columns": list(df.columns.astype(str)),
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    # Ensure read pointer at 0 for some containers
    await file.seek(0)
    df = _read_file_to_dataframe(file)
    session_id = str(uuid.uuid4())
    SESSION_STORE[session_id] = {
        "raw_df": df,
        "processed_df": None,
        "target": None,
        "problem_type": None,
        "pipeline": None,
        "model": None,
        "train": None,
        "test": None,
        "X_columns": None,
    }
    stats = _basic_stats(df)
    return {"session": session_id, **stats}


@app.get("/explore")
async def explore(session: str) -> Dict[str, Any]:
    state = SESSION_STORE.get(session)
    if not state:
        raise HTTPException(status_code=404, detail="Invalid session")
    df = state.get("processed_df") or state["raw_df"]
    return _basic_stats(df)


def _remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    clean_df = df.copy()
    for col in numeric_cols:
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (clean_df[col] >= lower) & (clean_df[col] <= upper)
        clean_df = clean_df[mask]
    return clean_df.reset_index(drop=True)


@app.post("/preprocess")
async def preprocess(session: str, options: PreprocessOptions) -> Dict[str, Any]:
    state = SESSION_STORE.get(session)
    if not state:
        raise HTTPException(status_code=404, detail="Invalid session")
    df = state["raw_df"].copy()

    if options.remove_outliers:
        df = _remove_outliers_iqr(df)

    # Separate columns by type
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if numeric_cols:
        num_steps = []
        if options.impute_missing:
            num_steps.append(("imputer", SimpleImputer(strategy="median")))
        if options.normalize:
            num_steps.append(("scaler", MinMaxScaler()))
        elif options.standardize:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=num_steps) if num_steps else "passthrough", numeric_cols))

    if categorical_cols and options.encode_categorical:
        cat_steps = []
        if options.impute_missing:
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore")))
        transformers.append(("cat", Pipeline(steps=cat_steps), categorical_cols))
    elif categorical_cols:
        transformers.append(("cat", "passthrough", categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Fit to compute resulting feature space
    X_transformed = preprocessor.fit_transform(df)
    state["processed_df"] = df  # Keep original rows filtered and imputed via pipeline during training
    state["preprocessor"] = preprocessor

    result = {
        "processed_rows": int(df.shape[0]),
        "final_features": int(X_transformed.shape[1]) if hasattr(X_transformed, "shape") else len(df.columns),
        "missing_ratio": 0.0 if options.impute_missing else round(float(df.isna().sum().sum() / (df.shape[0] * df.shape[1])), 4),
        "data_quality": 100,
    }
    return result


@app.post("/target/detect")
async def detect_target(session: str, payload: TargetRequest) -> Dict[str, Any]:
    state = SESSION_STORE.get(session)
    if not state:
        raise HTTPException(status_code=404, detail="Invalid session")
    df = state.get("processed_df") or state["raw_df"]
    if payload.target_variable not in df.columns:
        raise HTTPException(status_code=400, detail="Target variable not in dataset")

    target_series = df[payload.target_variable]
    n_unique = target_series.nunique(dropna=True)
    if target_series.dtype == "object" or n_unique <= 20:
        problem = "classification"
        distr = target_series.value_counts(dropna=True).to_dict()
        distribution = {str(k): int(v) for k, v in distr.items()}
    else:
        problem = "regression"
        distribution = None

    state["target"] = payload.target_variable
    state["problem_type"] = problem
    return {"problem_type": problem, "class_distribution": distribution}


@app.post("/train")
async def train(session: str, payload: TrainRequest) -> Dict[str, Any]:
    state = SESSION_STORE.get(session)
    if not state:
        raise HTTPException(status_code=404, detail="Invalid session")

    df = state.get("processed_df") or state["raw_df"]
    target = state.get("target")
    problem = state.get("problem_type")
    preprocessor = state.get("preprocessor")
    if target is None or problem is None:
        raise HTTPException(status_code=400, detail="Target not set. Call /target/detect first.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=payload.test_size, random_state=payload.random_state, stratify=y if problem == "classification" else None
    )

    if problem == "classification":
        model = RandomForestClassifier(random_state=payload.random_state, n_estimators=200)
    else:
        model = RandomForestRegressor(random_state=payload.random_state, n_estimators=200)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if problem == "classification":
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        try:
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)
                if y_prob.shape[1] == 2:
                    auc = float(roc_auc_score(y_test, y_prob[:, 1]))
                else:
                    auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))
            else:
                auc = None
        except Exception:
            auc = None
        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}
    else:
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        metrics = {"r2": r2, "mae": mae}

    state["pipeline"] = pipeline
    state["train"] = (X_train, y_train)
    state["test"] = (X_test, y_test)

    return {
        "trained": True,
        "problem_type": problem,
        "metrics": metrics,
        "test_size": payload.test_size,
        "model_name": payload.model_name,
    }


@app.get("/evaluate")
async def evaluate(session: str) -> Dict[str, Any]:
    state = SESSION_STORE.get(session)
    if not state or not state.get("pipeline"):
        raise HTTPException(status_code=400, detail="Model not trained")
    pipeline = state["pipeline"]
    X_test, y_test = state["test"]
    y_pred = pipeline.predict(X_test)

    problem = state["problem_type"]
    if problem == "classification":
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        try:
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)
                if y_prob.shape[1] == 2:
                    auc = float(roc_auc_score(y_test, y_prob[:, 1]))
                else:
                    auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))
            else:
                auc = None
        except Exception:
            auc = None
        return {"problem_type": problem, "metrics": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}}
    else:
        y_pred = pipeline.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        return {"problem_type": problem, "metrics": {"r2": r2, "mae": mae}}


@app.post("/predict")
async def predict(session: str, payload: PredictRequest) -> Dict[str, Any]:
    state = SESSION_STORE.get(session)
    if not state or not state.get("pipeline"):
        raise HTTPException(status_code=400, detail="Model not trained")
    pipeline = state["pipeline"]
    df = state.get("processed_df") or state["raw_df"]
    target = state["target"]
    feature_cols = [c for c in df.columns if c != target]

    # order features as training X columns
    record = {col: payload.features.get(col, None) for col in feature_cols}
    X_new = pd.DataFrame([record])
    pred = pipeline.predict(X_new)[0]

    response: Dict[str, Any] = {"prediction": pred if isinstance(pred, (int, float, str)) else str(pred)}
    if state["problem_type"] == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            prob = pipeline.predict_proba(X_new)[0]
            response["probabilities"] = [float(x) for x in prob]
        except Exception:
            pass
    return response


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "service": "DataFlow API"}


