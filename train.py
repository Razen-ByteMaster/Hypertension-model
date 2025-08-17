from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

DATA_PATH = r"C:\VScode-Project\GRAD PROJECT\New\Hypertension\hypertension_dataset.csv"
OUT_PATH = Path("best_model.joblib")
CV_SPLITS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

data_file = Path(DATA_PATH)
if not data_file.exists():
    print(f"ERROR: dataset not found at {data_file}", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(data_file)
print(f"Loaded data {data_file} shape={df.shape}")

TARGET_COL = None

if TARGET_COL:
    if TARGET_COL not in df.columns:
        print(f"ERROR: specified target '{TARGET_COL}' not in columns", file=sys.stderr)
        sys.exit(1)
    target_col = TARGET_COL
else:
    candidates = [
        c
        for c in df.columns
        if c.lower()
        in ("hypertension", "has_hypertension", "target", "label", "ht", "y")
    ]
    target_col = candidates[0] if candidates else df.columns[-1]

print(f"Using target column: '{target_col}'")
X = df.drop(columns=[target_col])
y = df[target_col]

le = None
if y.dtype == object or not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"Encoded target classes: {list(le.classes_)}")

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()
print(f"Numeric cols ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical cols ({len(categorical_cols)}): {categorical_cols}")

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop",
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
    try:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        cv_results[name] = {
            "mean_accuracy": float(scores.mean()),
            "std": float(scores.std()),
            "all": [float(s) for s in scores],
        }
        print(f"{name}: CV acc mean={scores.mean():.4f} std={scores.std():.4f}")
    except Exception as e:
        cv_results[name] = {"error": str(e)}
        print(f"{name}: ERROR during CV -> {e}")

with open("cv_results.json", "w", encoding="utf-8") as f:
    json.dump(cv_results, f, indent=2)
print("Saved cv_results.json")

valid = {k: v for k, v in cv_results.items() if "mean_accuracy" in v}
if not valid:
    print("No valid CV results. See cv_results.json for errors.", file=sys.stderr)
    sys.exit(1)

best_name = max(valid.items(), key=lambda kv: kv[1]["mean_accuracy"])[0]
best_clf = models[best_name]
print(
    f"Best model by CV accuracy: {best_name} (mean acc={valid[best_name]['mean_accuracy']:.4f})"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
best_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", best_clf)])
best_pipeline.fit(X_train, y_train)

y_pred = best_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy ({best_name}): {test_acc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

is_binary = len(np.unique(y)) == 2
if is_binary and hasattr(best_pipeline.named_steps["clf"], "predict_proba"):
    try:
        proba = best_pipeline.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, proba)
        print(f"ROC AUC: {roc:.4f}")
    except Exception as e:
        print(f"Could not compute ROC AUC: {e}")

joblib.dump(best_pipeline, OUT_PATH, compress=3)
print(f"\nSaved best pipeline to: {OUT_PATH}")

summary = {
    "data_path": str(data_file),
    "target": target_col,
    "best_model": best_name,
    "cv_mean_accuracy": valid[best_name]["mean_accuracy"],
    "cv_std": valid[best_name]["std"],
    "test_accuracy": float(test_acc),
}
with open("training_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("Saved training_summary.json")

try:
    import sklearn, numpy

    print(f"\nEnvironment: sklearn {sklearn.__version__}, numpy {numpy.__version__}")
except Exception:
    pass
