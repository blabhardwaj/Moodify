import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import joblib

CSV_PATH = "mood_color_dataset_extended.csv"
PCA_MODEL_PATH = "pca_model.pkl"
XGB_MODEL_PATH = "xgb_model.pkl"
N_COMPONENTS = 2  # Reduce to 2D RGB space

def train():
    df = pd.read_csv(CSV_PATH)
    
    X = df[["r", "g", "b"]].values
    y = df["label"].values

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # PCA
    pca = PCA(n_components=N_COMPONENTS)
    X_reduced = pca.fit_transform(X)

    # XGBoost
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_reduced, y_encoded)

    # Save models
    joblib.dump(pca, PCA_MODEL_PATH)
    joblib.dump(xgb, XGB_MODEL_PATH)

    print("✅ Training complete. Models saved:")
    print(f"  - PCA: {PCA_MODEL_PATH}")
    print(f"  - XGBoost: {XGB_MODEL_PATH}")

if __name__ == "__main__":
    train()
