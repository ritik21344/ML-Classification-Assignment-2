# Machine Learning Classification Assignment 2

## a. Problem statement
Build and evaluate five supervised classification algorithms on one common dataset, compare their predictive performance using six evaluation metrics, and provide a Streamlit application that evaluates labelled test CSV files using saved trained models.

## b. Dataset description
**Dataset:** Breast Cancer Wisconsin (Diagnostic), available through the scikit-learn dataset loader and originally associated with the UCI Machine Learning Repository.

- 569 instances and 30 numeric input features.
- Binary target: `0 = malignant`, `1 = benign`.
- 80% training data and 20% held-out testing data.
- Stratified split with `random_state=42`.
- StandardScaler is included in each saved pipeline so preprocessing is applied consistently at inference time.
- The included `test_data.csv` contains the held-out test features and the `target` column.

## c. GitHub Repository Link
Add your repository URL here after creating the GitHub repository: **[PASTE YOUR GITHUB REPOSITORY URL HERE]**

## d. Models used

The five classifiers are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine.

### Evaluation comparison

| Model                  |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-----------:|-------:|
| Logistic Regression    |     0.9825 | 0.9954 |      0.9861 |   0.9861 |     0.9861 | 0.9623 |
| Decision Tree          |     0.9035 | 0.9373 |      0.942  |   0.9028 |     0.922  | 0.7969 |
| K-Nearest Neighbors    |     0.9737 | 0.9884 |      0.96   |   1      |     0.9796 | 0.9442 |
| Gaussian Naive Bayes   |     0.9298 | 0.9868 |      0.9444 |   0.9444 |     0.9444 | 0.8492 |
| Support Vector Machine |     0.9825 | 0.995  |      0.9861 |   0.9861 |     0.9861 | 0.9623 |

Metrics are calculated on the same held-out test set. AUC uses the predicted probability for class 1 (benign).

### Observations on model performance

- **Logistic Regression:** Accuracy 0.9825, AUC 0.9954, F1 0.9861, and MCC 0.9623 on the held-out test split.
- **Decision Tree:** Accuracy 0.9035, AUC 0.9373, F1 0.9220, and MCC 0.7969 on the held-out test split.
- **K-Nearest Neighbors:** Accuracy 0.9737, AUC 0.9884, F1 0.9796, and MCC 0.9442 on the held-out test split.
- **Gaussian Naive Bayes:** Accuracy 0.9298, AUC 0.9868, F1 0.9444, and MCC 0.8492 on the held-out test split.
- **Support Vector Machine:** Accuracy 0.9825, AUC 0.9950, F1 0.9861, and MCC 0.9623 on the held-out test split.

**Overall winner for this dataset:** Logistic Regression, based on the highest F1 score in this experiment. This is a dataset- and split-specific result, not a claim that this model is universally best.

## Streamlit application
The app allows a user to upload a labelled CSV, select a saved classifier, view Accuracy, AUC, Precision, Recall, F1 and MCC, and inspect a confusion matrix and classification report. It also compares all five saved models on the uploaded test data. The app does not train models on uploaded test data.

**Live Streamlit App:** [PASTE YOUR DEPLOYED STREAMLIT URL HERE]

## Repository contents
- `app.py` — Streamlit application.
- `train_models.py` — reproducible model training and evaluation script.
- `requirements.txt` — Python dependencies.
- `test_data.csv` — held-out test dataset with target.
- `model/` — saved model pipelines.
- `feature_columns.json` — required input feature names.
- `results/model_comparison.csv` — training/test split comparison metrics.

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to reproduce training
```bash
python train_models.py
```

## Target encoding
- `0`: malignant
- `1`: benign
