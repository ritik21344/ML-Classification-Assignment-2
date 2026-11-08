"""Train five classification pipelines and create the test CSV and comparison table."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).parent
(ROOT / "model").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)
dataset = load_breast_cancer(as_frame=True)
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=3000, random_state=42))]),
    "Decision Tree": Pipeline([("scaler", StandardScaler()), ("classifier", DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42))]),
    "K-Nearest Neighbors": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=7))]),
    "Gaussian Naive Bayes": Pipeline([("scaler", StandardScaler()), ("classifier", GaussianNB())]),
    "Support Vector Machine": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))]),
}
files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
rows = []
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    joblib.dump(estimator, ROOT / "model" / files[name])
    predicted = estimator.predict(X_test)
    probability = estimator.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "AUC": roc_auc_score(y_test, probability),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1 Score": f1_score(y_test, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predicted),
    })
pd.DataFrame(rows).to_csv(ROOT / "results" / "model_comparison.csv", index=False)
test = X_test.copy()
test["target"] = y_test
test.to_csv(ROOT / "test_data.csv", index=False)
(ROOT / "feature_columns.json").write_text(json.dumps(list(X.columns), indent=2))
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("Training complete. Models, test_data.csv and results are saved.")
