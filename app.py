import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classification Model Evaluator", page_icon="📊", layout="wide")
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET = "target"

st.title("📊 Machine Learning Classification Dashboard")
st.write("Compare five trained classification models on an uploaded, labelled test CSV.")
st.info("Upload a CSV containing the 30 input features and a target column named `target`. The app evaluates saved models; it does not retrain them.")

with st.sidebar:
    st.header("Evaluation settings")
    selected_model = st.selectbox("Choose a classification model", list(MODEL_FILES.keys()))
    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

st.subheader("Test dataset")
if uploaded_file is None:
    default_path = ROOT / "test_data.csv"
    if default_path.exists():
        data = pd.read_csv(default_path)
        st.caption("Showing the included hold-out test_data.csv. You can upload another CSV using the sidebar.")
    else:
        st.warning("Upload a labelled test CSV to begin.")
        st.stop()
else:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

if TARGET not in data.columns:
    st.error("The uploaded CSV must contain a target column named 'target'.")
    st.stop()

feature_columns = [c for c in data.columns if c != TARGET]
try:
    model_path = MODEL_DIR / MODEL_FILES[selected_model]
    model = joblib.load(model_path)
    expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    # The training pipeline may not expose feature_names_in_ consistently; use saved metadata below.
    metadata_path = ROOT / "feature_columns.json"
    expected_features = json.loads(metadata_path.read_text()) if metadata_path.exists() else feature_columns
    missing = [c for c in expected_features if c not in data.columns]
    extra = [c for c in feature_columns if c not in expected_features]
    if missing:
        st.error("Missing required feature columns: " + ", ".join(missing))
        st.stop()
    if extra:
        st.warning("Extra columns will be ignored: " + ", ".join(extra))
    X_eval = data[expected_features].apply(pd.to_numeric, errors="coerce")
    if X_eval.isna().any().any():
        st.error("Feature columns contain missing or non-numeric values. Please clean the CSV and upload again.")
        st.stop()
    y_true = pd.to_numeric(data[TARGET], errors="raise").astype(int)
    if not set(y_true.unique()).issubset({0, 1}):
        st.error("The target column must use 0 and 1 labels (0 = malignant, 1 = benign).")
        st.stop()
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_prob),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}
st.subheader(f"Results: {selected_model}")
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("#### Classification report")
    report = classification_report(y_true, y_pred, target_names=["Malignant (0)", "Benign (1)"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.caption(f"Rows evaluated: {len(data)} | Features used: {len(expected_features)} | Target: {TARGET}")
st.subheader("Compare all five models on this uploaded dataset")
comparison = []
for model_name, filename in MODEL_FILES.items():
    candidate = joblib.load(MODEL_DIR / filename)
    p = candidate.predict(X_eval)
    pp = candidate.predict_proba(X_eval)[:, 1]
    comparison.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, p),
        "AUC": roc_auc_score(y_true, pp),
        "Precision": precision_score(y_true, p, zero_division=0),
        "Recall": recall_score(y_true, p, zero_division=0),
        "F1 Score": f1_score(y_true, p, zero_division=0),
        "MCC": matthews_corrcoef(y_true, p),
    })
st.dataframe(pd.DataFrame(comparison).set_index("Model").round(4), use_container_width=True)
