import os
import sys
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve


try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = "loan_approval_dataset.csv"  
TARGET_NAME = None  
SAVE_MODEL_PATH = "loan_logreg_model.pkl"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Can't find dataset at {DATA_PATH}. Put the CSV there or change DATA_PATH.")

df = pd.read_csv(DATA_PATH, dtype=str) 
print("Original shape:", df.shape)
print("Raw columns:", df.columns.tolist())


df.columns = df.columns.str.strip()


for col in df.columns:
    if df[col].dtype == object or df[col].dtype == 'string':
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'': np.nan, 'nan': np.nan})

print("Cleaned columns:", df.columns.tolist())


if TARGET_NAME is None:
    possible_targets = ["loan_status", "Loan_Status", "LoanStatus", "loan_status", "target", "approved", "is_approved"]
    TARGET_NAME = None
    for t in possible_targets:
        if t in df.columns:
            TARGET_NAME = t
            break
    if TARGET_NAME is None:
        TARGET_NAME = df.columns[-1]
        print(f"No common target name found; using last column as target: '{TARGET_NAME}'")
    else:
        print(f"Detected target column: '{TARGET_NAME}'")
else:
    if TARGET_NAME not in df.columns:
        raise ValueError(f"TARGET_NAME '{TARGET_NAME}' not found in columns.")


df[TARGET_NAME] = df[TARGET_NAME].astype(str).str.strip()


target_lower = df[TARGET_NAME].str.lower()
mapping = {
    "approved": 1, "yes": 1, "y": 1, "1": 1,
    "rejected": 0, "no": 0, "n": 0, "0": 0,
    "denied": 0
}

if target_lower.dropna().isin(mapping.keys()).all():
    df[TARGET_NAME] = target_lower.map(mapping).astype(int)
    print("Mapped target using common mapping.")
else:

    df[TARGET_NAME], uniques = pd.factorize(df[TARGET_NAME].fillna("MISSING"))
    print("Factorized target; mappings:", dict(enumerate(uniques)))


print("Target value counts:\n", pd.Series(df[TARGET_NAME]).value_counts())


for col in df.columns:
    if col == TARGET_NAME:
        continue
    conv = pd.to_numeric(df[col], errors='coerce')

    if conv.notna().sum() >= 0.5 * len(conv):
        df[col] = conv

X = df.drop(columns=[TARGET_NAME])
y = df[TARGET_NAME].astype(int)


numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

print("Numeric cols:", numeric_cols)
print("Categorical cols:", cat_cols)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=RANDOM_SEED,
                                                    stratify=y)
print("\nTrain target distribution:\n", y_train.value_counts(normalize=True))
print("Test target distribution:\n", y_test.value_counts(normalize=True))

ohe_params = {"handle_unknown": "ignore"}
sig = inspect.signature(OneHotEncoder)
if "sparse_output" in sig.parameters:
    ohe_params["sparse_output"] = False
elif "sparse" in sig.parameters:
    ohe_params["sparse"] = False
ohe = OneHotEncoder(**ohe_params)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", ohe)
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, cat_cols)
], remainder="drop")  

if IMBLEARN_AVAILABLE:
    print("imblearn detected — will use SMOTE after preprocessing.")
    pipe = ImbPipeline(steps=[
        ("preproc", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
    ])
else:
    print("imblearn NOT installed — falling back to class_weight='balanced' (no SMOTE).")
    pipe = Pipeline(steps=[
        ("preproc", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced"))
    ])

param_grid = {
    "clf__C": [0.01, 0.1, 1.0, 5.0],
    "clf__penalty": ["l2"],
    "clf__solver": ["liblinear", "saga"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

grid = GridSearchCV(pipe, param_grid, scoring="f1", n_jobs=-1, cv=cv, verbose=1, error_score="raise")

print("\nFitting pipeline (this may take a while)...")
grid.fit(X_train, y_train)

print("\nBest params:", grid.best_params_)
print("Best CV F1-score:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = None
if hasattr(best_model, "predict_proba"):
    try:
        y_proba = best_model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

print("\nClassification report (test):")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

if y_proba is not None:
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC (test): {roc_auc:.4f}")
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

joblib.dump(best_model, SAVE_MODEL_PATH)
print(f"Saved trained model to {SAVE_MODEL_PATH}")

try:
    preproc = best_model.named_steps["preproc"]
    feature_names = []
    if numeric_cols:
        feature_names.extend(numeric_cols)
    if cat_cols:
        ohe_step = preproc.named_transformers_["cat"].named_steps["onehot"]
        if hasattr(ohe_step, "get_feature_names_out"):
            cat_ohe_names = ohe_step.get_feature_names_out(cat_cols).tolist()
        else:
            categories = preproc.named_transformers_["cat"].named_steps["onehot"].categories_
            cat_ohe_names = []
            for col, cats in zip(cat_cols, categories):
                cat_ohe_names += [f"{col}_{c}" for c in cats]
        feature_names.extend(cat_ohe_names)
    print("\nFinal feature vector length after preprocessing:", len(feature_names))
except Exception as e:
    print("Could not list feature names:", e)

print("\nDone.")
