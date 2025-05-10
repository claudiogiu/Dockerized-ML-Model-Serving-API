import os
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from load_data import load_data
import warnings

warnings.filterwarnings("ignore")

def test_model(X_test: pd.DataFrame, y_test: pd.DataFrame):
    """Load saved models (Scaler, PCA, SVM), transform features, and evaluate performance."""

    models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    # Load the SVM model
    svc_model_path = os.path.join(models_path, "svc_model.pkl")
    if not os.path.exists(svc_model_path):
        raise FileNotFoundError(f"SVM model not found. Expected at: {svc_model_path}")

    with open(svc_model_path, "rb") as f:
        svc = pickle.load(f)

    # Load the PCA model
    pca_model_path = os.path.join(models_path, "pca_model.pkl")
    if not os.path.exists(pca_model_path):
        raise FileNotFoundError(f"PCA model not found. Expected at: {pca_model_path}")

    with open(pca_model_path, "rb") as f:
        pca = pickle.load(f)

    # Load the Scaler model
    scaler_model_path = os.path.join(models_path, "scaler.pkl")
    if not os.path.exists(scaler_model_path):
        raise FileNotFoundError(f"Scaler model not found. Expected at: {scaler_model_path}")

    with open(scaler_model_path, "rb") as f:
        scaler = pickle.load(f)

    # Standardization
    X_test_scaled = scaler.transform(X_test)

    # PCA
    X_test_transformed = pca.transform(X_test_scaled)

    y_test = y_test.replace({"Siit_Pistachio": 1, "Kirmizi_Pistachio": 0})

    # Predict using the SVM model
    y_pred = svc.predict(X_test_transformed)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    sensitivity = recall_score(y_test, y_pred, pos_label=1, average="binary") * 100
    specificity = recall_score(y_test, y_pred, pos_label=0, average="binary") * 100
    auc = roc_auc_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100


    print("Model Evaluation Metrics:")
    print(f"Accuracy Score(%): {accuracy:.2f}")
    print(f"Precision Score (%): {precision:.2f}")
    print(f"Sensitivity Score (%): {sensitivity:.2f}")
    print(f"Specificity Score (%): {specificity:.2f}")
    print(f"AUC Score (%): {auc:.2f}") 
    print(f"F1 Score (%): {f1:.2f}") 

if __name__ == "__main__":

    X_test = load_data("X_test.csv", "processed")
    y_test = load_data("y_test.csv", "processed")

    test_model(X_test, y_test)