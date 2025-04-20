# ml_pipeline.py

import numpy as np
import cv2
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import joblib
from PIL import Image

class MoodClassifierPipeline:
    def __init__(self, pca_path='pca_model.pkl', xgb_path='xgb_model.pkl'):
        try:
            self.pca = joblib.load(pca_path)
            self.xgb = joblib.load(xgb_path)
            self.loaded = True
        except:
            print("⚠️ Could not load PCA/XGBoost. Falling back to random mood.")
            self.loaded = False

        self.emotions = ["happy", "sad", "angry", "calm", "excited"]

    def extract_features(self, image_data):
        """Extract basic color + texture features."""
        img = Image.fromarray(image_data.astype('uint8'), mode='RGBA').convert('RGB')
        img = img.resize((64, 64))
        arr = np.array(img)

        # Flatten RGB values as features
        features = arr.reshape(-1, 3).mean(axis=0)  # Average color
        return features.reshape(1, -1)

    def classify_mood(self, image_data):
        if not self.loaded:
            import random
            return random.choice(self.emotions)

        try:
            features = self.extract_features(image_data)
            reduced = self.pca.transform(features)
            prediction = self.xgb.predict(reduced)
            return self.emotions[int(prediction[0])]
        except Exception as e:
            print("Classification error:", e)
            import random
            return random.choice(self.emotions)
