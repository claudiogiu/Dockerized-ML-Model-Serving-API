import os
import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from load_data import load_data
import warnings

warnings.filterwarnings("ignore")

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Load saved scaler and PCA, transform features, train the SVC model, and save it."""

    models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    y_train = y_train.replace({"Siit_Pistachio": 1, "Kirmizi_Pistachio": 0})

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train.to_numpy().ravel())

    # Load the saved scaler
    scaler_model_path = os.path.join(models_path, "scaler.pkl")
    if not os.path.exists(scaler_model_path):
        raise FileNotFoundError(f"Scaler model not found. Expected at: {scaler_model_path}")

    with open(scaler_model_path, "rb") as f:
        scaler = pickle.load(f)

    # Standardization using the saved model
    X_train_scaled = scaler.transform(X_train_resampled)

    # Load the saved PCA model
    pca_model_path = os.path.join(models_path, "pca_model.pkl")
    if not os.path.exists(pca_model_path):
        raise FileNotFoundError(f"PCA model not found. Expected at: {pca_model_path}")

    with open(pca_model_path, "rb") as f:
        pca = pickle.load(f)

    # Apply PCA
    X_train_transformed = pca.transform(X_train_scaled)

    # Train the SVM model
    svc = SVC(C=0.1, kernel='rbf', gamma='scale')
    svc.fit(X_train_transformed, y_train_resampled)

    # Save the trained SVM model
    svc_model_path = os.path.join(models_path, "svc_model.pkl")
    with open(svc_model_path, "wb") as f:
        pickle.dump(svc, f)

    print("SVC model trained and successfully saved in the 'models' folder.")

if __name__ == "__main__":
    
    X_train = load_data("X_train.csv", "processed")
    y_train = load_data("y_train.csv", "processed")

    train_model(X_train, y_train)