# ml_pipeline
import numpy as np
import cv2
import joblib
from PIL import Image
import logging
import os
import json
import glob

class MoodClassifierPipeline:
    def __init__(self, model_dir='./models', model_version='1.0.0', timestamp=None,
                 pca_model_name='pca_model.pkl', xgb_model_name='xgb_model.pkl',
                 scaler_model_name='scaler_model.pkl'):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Store parameters
        self.model_dir = model_dir
        self.model_version = model_version
        self.timestamp = timestamp
        self.default_mood = "calm"
        self.confidence_threshold = 0.3  # default, will be updated from metadata if available

        # If timestamp is provided, use specific model version, otherwise find latest
        if timestamp:
            self.model_prefix = f"{model_version}_{timestamp}_"
            self.logger.info(f"Using specific model version: {model_version}, timestamp: {timestamp}")
        else:
            self.model_prefix = self._find_latest_model_prefix(model_version)
            self.logger.info(f"Using latest model with prefix: {self.model_prefix}")
            
        # Construct full paths
        pca_path = os.path.join(model_dir, f"{self.model_prefix}{pca_model_name}")
        xgb_path = os.path.join(model_dir, f"{self.model_prefix}{xgb_model_name}")
        scaler_path = os.path.join(model_dir, f"{self.model_prefix}{scaler_model_name}")
        
        # Load models
        self.loaded = self._load_models(pca_path, xgb_path, scaler_path)
        
        # Load metadata and class labels
        if self.loaded:
            self._load_metadata()
            self._load_class_labels()

    def _find_latest_model_prefix(self, version):
        """Find the latest timestamp for a given model version"""
        try:
            files = os.listdir(self.model_dir)
            matching_files = [f for f in files if f.startswith(version) and f.endswith('.pkl')]
            
            if not matching_files:
                self.logger.warning(f"No models found for version {version}, falling back to default naming")
                return ""
                
            # Sort by timestamp (which follows the version in the filename)
            matching_files.sort(reverse=True)
            # Extract prefix from first file (most recent)
            prefix = "_".join(matching_files[0].split("_")[:2]) + "_"
            
            self.logger.info(f"Found latest model: {prefix}")
            return prefix
        except Exception as e:
            self.logger.error(f"Error finding latest model: {str(e)}")
            return ""

    def _load_models(self, pca_path, xgb_path, scaler_path):
        """Load all required models"""
        try:
            # Load XGBoost model (required)
            if not os.path.exists(xgb_path):
                self.logger.warning(f"XGB model file not found at {xgb_path}")
                return False
            
            self.xgb = joblib.load(xgb_path)
            self.logger.info(f"XGBoost model loaded from {xgb_path}")

            # Load PCA model (optional)
            if os.path.exists(pca_path):
                self.pca = joblib.load(pca_path)
                self.logger.info(f"PCA model loaded from {pca_path}")
            else:
                self.pca = None
                self.logger.info("No PCA model found, features will be used directly")

            # Load scaler (required)
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.has_scaler = True
                self.logger.info(f"Scaler loaded from {scaler_path}")
            else:
                self.logger.warning(f"Scaler not found at {scaler_path}, features won't be standardized")
                self.has_scaler = False

            self.logger.info("Models loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            return False

    def _load_metadata(self):
        """Load model metadata for validation and information"""
        try:
            metadata_path = os.path.join(self.model_dir, f"{self.model_prefix}metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.logger.info(f"Loaded model metadata: v{self.metadata['version']}, " 
                                    f"accuracy: {self.metadata['performance']['accuracy']:.4f}")
                    
                    # Validate feature count
                    if 'training_config' in self.metadata and 'feature_count' in self.metadata['training_config']:
                        self.expected_feature_count = self.metadata['training_config']['feature_count']
                    else:
                        self.expected_feature_count = 36  # default (12 basic + 24 histogram)
                    
                    # Store confidence threshold if available
                    if 'confidence_threshold' in self.metadata:
                        self.confidence_threshold = self.metadata['confidence_threshold']
                
                return True
            else:
                self.logger.warning(f"Metadata file not found: {metadata_path}")
                self.metadata = None
                self.expected_feature_count = 36  # default
                return False
        except Exception as e:
            self.logger.error(f"Error loading metadata: {str(e)}")
            self.metadata = None
            return False

    def _load_class_labels(self):
        """Load class labels from the mapping file"""
        try:
            mapping_path = os.path.join(self.model_dir, f"{self.model_prefix}class_mapping.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    class_mapping = json.load(f)
                    # Convert from dict to ordered list based on indices
                    self.emotions = [label for idx, label in sorted([(int(v), k) for k, v in class_mapping.items()])]
                    self.logger.info(f"Loaded {len(self.emotions)} class labels: {self.emotions}")
                    return True
            else:
                self.logger.warning(f"Class mapping file not found: {mapping_path}")
                # Fallback to default labels
                self.emotions = ["happy", "sad", "angry", "calm", "excited"]
                self.logger.info(f"Using default emotion labels: {self.emotions}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading class labels: {str(e)}")
            self.emotions = ["happy", "sad", "angry", "calm", "excited"]
            return False

    def preprocess_image(self, image_data):
        """Preprocess image data to handle different formats"""
        try:
            if isinstance(image_data, str):
                if os.path.exists(image_data):
                    img = Image.open(image_data).convert('RGB')
                    return np.array(img)
                else:
                    raise FileNotFoundError(f"Image file not found: {image_data}")

            if isinstance(image_data, np.ndarray):
                if len(image_data.shape) == 2:  # Grayscale
                    self.logger.info("Converting grayscale to RGB")
                    return cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
                elif len(image_data.shape) == 3:
                    if image_data.shape[2] == 4:  # RGBA
                        self.logger.info("Converting RGBA to RGB")
                        img = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('RGB')
                        return np.array(img)
                    elif image_data.shape[2] == 3:  # RGB
                        return image_data
                    else:
                        raise ValueError(f"Unsupported number of channels: {image_data.shape[2]}")

            raise TypeError(f"Unsupported image data type: {type(image_data)}")

        except Exception as e:
            self.logger.error(f"Image preprocessing error: {str(e)}")
            return None

    def extract_features(self, image_data):
        """Extract color features from image data to match training pipeline"""
        try:
            arr = self.preprocess_image(image_data)
            if arr is None:
                raise ValueError("Image preprocessing failed")

            # Basic color statistics (12 features)
            basic_features = []
            for i, channel_name in enumerate(['r', 'g', 'b']):
                channel = arr[:, :, i]
                basic_features.extend([
                    channel.mean(),  # mean
                    channel.std(),   # std
                    channel.min(),   # min
                    channel.max()    # max
                ])

            # Histogram features (24 features: 8 bins × 3 channels)
            hist_features = []
            for i in range(3):
                # Use 8 bins to match training pipeline
                hist = cv2.calcHist([arr], [i], None, [8], [0, 256])
                # Normalize histogram
                hist = hist.flatten() / hist.sum()
                hist_features.extend(hist)

            # Combine all features (36 total)
            all_features = np.array(basic_features + hist_features).reshape(1, -1)
            
            # Validate feature count if we have metadata
            if hasattr(self, 'expected_feature_count') and all_features.shape[1] != self.expected_feature_count:
                self.logger.warning(f"Feature count mismatch: extracted {all_features.shape[1]}, expected {self.expected_feature_count}")
            
            self.logger.debug(f"Extracted {all_features.shape[1]} features: 12 basic + 24 histogram")
            return all_features

        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            return None

    def _apply_pca_if_needed(self, features):
        """Apply PCA transformation only if it was used in training"""
        try:
            if not hasattr(self, 'pca') or self.pca is None:
                self.logger.debug("No PCA model available, using features directly")
                return features
                
            if self.metadata and 'training_config' in self.metadata and not self.metadata['training_config'].get('use_pca', True):
                self.logger.debug("PCA was disabled in training, using features directly")
                return features
                
            reduced = self.pca.transform(features)
            self.logger.debug(f"Applied PCA: {features.shape[1]} → {reduced.shape[1]} features")
            return reduced
        except Exception as e:
            self.logger.error(f"PCA transformation error: {str(e)}")
            return features

    def get_prediction_confidence(self, probabilities):
        """Calculate prediction confidence based on probability distribution"""
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) > 1:
            # Confidence is the difference between top two probabilities
            confidence = sorted_probs[0] - sorted_probs[1]
        else:
            confidence = sorted_probs[0]
        return confidence

    def classify_mood(self, image_data):
        """Classify the mood based on image data"""
        if not self.loaded:
            self.logger.warning("Models not loaded, returning default mood")
            return {"mood": self.default_mood, "confidence": 0.0}

        try:
            features = self.extract_features(image_data)
            if features is None:
                raise ValueError("Feature extraction failed")

            if self.has_scaler:
                features = self.scaler.transform(features)

            # Apply PCA if needed
            reduced = self._apply_pca_if_needed(features)

            # Get prediction and confidence
            if hasattr(self.xgb, 'predict_proba'):
                probabilities = self.xgb.predict_proba(reduced)[0]
                prediction_idx = np.argmax(probabilities)
                confidence = self.get_prediction_confidence(probabilities)
                
                self.logger.info(f"Prediction confidence: {confidence:.2f}")

                if confidence < self.confidence_threshold:
                    self.logger.warning(f"Low confidence prediction ({confidence:.2f}), using default mood")
                    return {"mood": self.default_mood, "confidence": confidence}
            else:
                prediction_idx = int(self.xgb.predict(reduced)[0])
                confidence = 1.0  # No confidence available

            predicted_mood = self.emotions[prediction_idx]
            self.logger.info(f"Predicted mood: {predicted_mood} with confidence {confidence:.2f}")
            return {"mood": predicted_mood, "confidence": confidence}

        except Exception as e:
            self.logger.error(f"Classification error: {str(e)}")
            return {"mood": self.default_mood, "confidence": 0.0}

    def classify_batch(self, image_data_list):
        """Process a batch of images for better efficiency"""
        results = []
        for image_data in image_data_list:
            results.append(self.classify_mood(image_data))
        return results

    def evaluate_model(self, test_images, test_labels):
        """Evaluate model performance on test data"""
        if not self.loaded:
            self.logger.error("Models not loaded, cannot evaluate")
            return None

        predictions = []
        confidences = []
        for image in test_images:
            result = self.classify_mood(image)
            predictions.append(result["mood"])
            confidences.append(result["confidence"])

        correct = sum(1 for pred, label in zip(predictions, test_labels) if pred == label)
        accuracy = correct / len(test_labels)

        class_metrics = {}
        for emotion in self.emotions:
            true_positives = sum(1 for pred, label in zip(predictions, test_labels)
                                 if pred == emotion and label == emotion)
            false_positives = sum(1 for pred, label in zip(predictions, test_labels)
                                  if pred == emotion and label != emotion)
            false_negatives = sum(1 for pred, label in zip(predictions, test_labels)
                                  if pred != emotion and label == emotion)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_metrics[emotion] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'accuracy': accuracy,
            'class_metrics': class_metrics,
            'avg_confidence': avg_confidence,
            'model_version': self.model_version,
            'timestamp': self.timestamp or self.model_prefix.split('_')[1] if '_' in self.model_prefix else 'unknown'
        }

    def get_model_info(self):
        """Return information about the loaded model"""
        if not self.loaded:
            return {"status": "not_loaded"}
            
        info = {
            "status": "loaded",
            "version": self.model_version,
            "timestamp": self.timestamp or self.model_prefix.split('_')[1] if '_' in self.model_prefix else 'unknown',
            "classes": self.emotions,
            "default_mood": self.default_mood,
            "confidence_threshold": self.confidence_threshold
        }
        
        if self.metadata:
            info["performance"] = self.metadata.get("performance", {})
            info["training_config"] = self.metadata.get("training_config", {})
            
        return info
