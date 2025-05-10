import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from load_data import load_data
import warnings

warnings.filterwarnings("ignore")

def split_data(X: pd.DataFrame, y: pd.DataFrame):
    """Split X and y into train/test (80-20) and save the data in the 'data/processed' folder."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    processed_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    os.makedirs(processed_path, exist_ok=True)

    X_train.to_csv(os.path.join(processed_path, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(processed_path, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_path, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(processed_path, "y_test.csv"), index=False)

    print("Train/Test split saved in 'data/processed'.")

def preprocess(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Apply SMOTE, standardize features, apply PCA, and save both the scaler and PCA model."""

    models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_path, exist_ok=True)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)

    # Save the Scaler Model
    scaler_model_path = os.path.join(models_path, "scaler.pkl")
    with open(scaler_model_path, "wb") as f:
        pickle.dump(scaler, f)

    # PCA
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Save the PCA Model
    pca_model_path = os.path.join(models_path, "pca_model.pkl")
    with open(pca_model_path, "wb") as f:
        pickle.dump(pca, f)

    print("scaler.pkl and pca_model.pkl correctly saved in 'models'.")

if __name__ == "__main__":
    
    dataset_name = "Pistachio_16_Features_Dataset.csv"
    df = load_data(dataset_name, "raw")  

    df = df.iloc[:, 1:]  

    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    split_data(X, y)


    X_train = load_data("X_train.csv", "processed")
    y_train = load_data("y_train.csv", "processed")

    preprocess(X_train, y_train)