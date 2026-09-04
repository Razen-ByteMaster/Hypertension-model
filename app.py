from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

app = FastAPI(title="Hypertension Train & Predict API")

_BASE_DIR = Path(__file__).parent
DEFAULT_DATA_PATH = str(_BASE_DIR / "hypertension_dataset.csv")
DEFAULT_OUT_MODEL = _BASE_DIR / "best_model.joblib"
CV_SPLITS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

NUMERIC_COLS = ["Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI"]
CATEGORICAL_COLS = [
    "BP_History",
    "Medication",
    "Family_History",
    "Exercise_Level",
    "Smoking_Status",
]
TARGET_DEFAULT = "Has_Hypertension"

_loaded_pipeline = None
_target_classes = None
_label_encoder = None
_saved_model_path = None


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def fit_label_encoder_if_needed(y):
    global _label_encoder
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        _label_encoder = le
        return y_enc
    return y


def target_classes_from(y) -> list:
    """JSON-safe label list: encoder classes if used, else sorted unique values."""
    global _label_encoder
    if _label_encoder is not None:
        return [str(c) for c in _label_encoder.classes_]
    return sorted({int(v) for v in np.asarray(y).ravel()})


def pack_artifact(pipeline: Pipeline, target_classes: list, best_name: str) -> Dict[str, Any]:
    return {
        "pipeline": pipeline,
        "target_classes": target_classes,
        "best_model": best_name,
    }


def unpack_artifact(obj):
    """Accept current dict artifacts and legacy bare-pipeline files."""
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"], obj.get("target_classes")
    return obj, None


def save_pipeline(artifact: Dict[str, Any], out_path: Path):
    joblib.dump(artifact, out_path, compress=3)
    return str(out_path.resolve())


def load_pipeline(path: Path):
    global _loaded_pipeline, _target_classes, _saved_model_path
    if _loaded_pipeline is None or (
        _saved_model_path and str(path.resolve()) != str(_saved_model_path)
    ):
        _loaded_pipeline, _target_classes = unpack_artifact(joblib.load(path))
        _saved_model_path = str(path.resolve())
    return _loaded_pipeline


def safe_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, zero_division=0, output_dict=True)


class TrainRequest(BaseModel):
    data_path: Optional[str] = None
    target_col: Optional[str] = TARGET_DEFAULT
    out_model: Optional[str] = None
    cv_splits: Optional[int] = CV_SPLITS
    test_size: Optional[float] = TEST_SIZE
    random_state: Optional[int] = RANDOM_STATE


class PredictRequest(BaseModel):
    Age: float
    Salt_Intake: float
    Stress_Score: float
    Sleep_Duration: float
    BMI: float
    BP_History: str
    Medication: str
    Family_History: str
    Exercise_Level: str
    Smoking_Status: str


@app.post("/train")
def train(req: TrainRequest):
    data_path = req.data_path or DEFAULT_DATA_PATH
    data_file = Path(data_path)
    if not data_file.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {data_file}")

    df = pd.read_csv(data_file)
    target_col = req.target_col or TARGET_DEFAULT
    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' not found in CSV columns.",
        )

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]
    y = fit_label_encoder_if_needed(y_raw)

    missing_numeric = [c for c in NUMERIC_COLS if c not in X.columns]
    missing_cat = [c for c in CATEGORICAL_COLS if c not in X.columns]
    if missing_numeric or missing_cat:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "CSV columns do not match expected features.",
                "missing_numeric": missing_numeric,
                "missing_categorical": missing_cat,
            },
        )

    preprocessor = build_preprocessor()

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=req.random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=req.random_state
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=req.random_state),
    }

    cv = StratifiedKFold(
        n_splits=req.cv_splits, shuffle=True, random_state=req.random_state
    )
    cv_results = {}
    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
            cv_results[name] = {
                "mean_accuracy": float(scores.mean()),
                "std": float(scores.std()),
                "folds": [float(s) for s in scores],
            }
        except Exception as e:
            cv_results[name] = {"error": str(e)}

    valid = {k: v for k, v in cv_results.items() if "mean_accuracy" in v}
    if not valid:
        raise HTTPException(
            status_code=500,
            detail={"error": "No successful CV results", "cv_results": cv_results},
        )

    best_name = max(valid.items(), key=lambda kv: kv[1]["mean_accuracy"])[0]
    best_clf = models[best_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, stratify=y, random_state=req.random_state
    )
    best_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", best_clf)])
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))
    report = safe_classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    out_model_path = Path(req.out_model) if req.out_model else DEFAULT_OUT_MODEL
    target_classes = target_classes_from(y)
    saved_path = save_pipeline(
        pack_artifact(best_pipeline, target_classes, best_name), out_model_path
    )

    global _loaded_pipeline, _target_classes
    _loaded_pipeline = best_pipeline
    _target_classes = target_classes

    summary = {
        "data_path": str(data_file.resolve()),
        "target_col": target_col,
        "best_model": best_name,
        "target_classes": target_classes,
        "cv_results": cv_results,
        "test_accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "saved_model_path": saved_path,
    }
    return summary


@app.post("/predict")
def predict(payload: PredictRequest):
    out_path = DEFAULT_OUT_MODEL
    if not Path(out_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"No trained model found at {out_path}. Call /train first.",
        )

    pipeline = load_pipeline(Path(out_path))

    row = {k: getattr(payload, k) for k in (NUMERIC_COLS + CATEGORICAL_COLS)}
    df_row = pd.DataFrame([row])

    try:
        pred = pipeline.predict(df_row)[0]
        prob = None
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            prob = float(pipeline.predict_proba(df_row).max())
        # Prefer classes stored in the artifact (works on fresh loads);
        # fall back to the in-memory encoder, then the raw value.
        if _target_classes:
            pred_label = _target_classes[int(pred)]
        elif _label_encoder is not None:
            pred_label = str(_label_encoder.inverse_transform([int(pred)])[0])
        else:
            pred_label = int(pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"prediction": pred_label, "probability": prob}


@app.get("/model")
def model_info():
    p = Path(DEFAULT_OUT_MODEL)
    if not p.exists():
        return {"exists": False, "path": str(p.resolve())}
    info = {"exists": True, "path": str(p.resolve())}
    try:
        pipeline = load_pipeline(p)
        clf = pipeline.named_steps.get("clf")
        info["estimator"] = type(clf).__name__
    except Exception:
        info["estimator"] = "unknown (could not load)"
    return info
