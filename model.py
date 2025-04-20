import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

class EmotionClassifier:
    def __init__(self, model_path='emotion_model.h5'):
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.loaded = True
        except:
            print(f"Could not load model from {model_path}. Using fallback classifier.")
            self.loaded = False
            
        self.emotions = ["happy", "sad", "angry", "calm", "excited"]
    
    def preprocess_image(self, image_data):
        """Convert canvas image data to model input format"""
        # Convert RGBA to RGB
        img = Image.fromarray(image_data.astype('uint8'), mode='RGBA')
        img = img.convert('RGB')
        
        # Resize to model input size (adjust based on your model requirements)
        img = img.resize((48, 48))
        
        # Convert to grayscale if your model expects single channel
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Normalize pixel values
        normalized = gray / 255.0
        
        # Reshape for model input (adjust dimensions based on your model)
        input_data = normalized.reshape(1, 48, 48, 1)
        
        return input_data
    
    def classify_mood(self, image_data):
        """Predict emotion from image data"""
        if not self.loaded or image_data is None:
            # Fallback to random selection if model not loaded
            import random
            return random.choice(self.emotions)
        
        try:
            # Preprocess the image
            input_data = self.preprocess_image(image_data)
            
            # Make prediction
            prediction = self.model.predict(input_data)
            emotion_index = np.argmax(prediction)
            
            # Map index to emotion (adjust based on your model's output classes)
            # This assumes your model outputs in the same order as self.emotions
            return self.emotions[emotion_index % len(self.emotions)]
        except Exception as e:
            print(f"Error in classification: {e}")
            # Fallback to random selection
            import random
            return random.choice(self.emotions)


# Function to create and train a simple emotion detection model
def create_emotion_model():
    """Create a simple CNN model for emotion detection from doodles"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')  # 5 emotions
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Note: You would need to train this model with a dataset of doodles labeled with emotions
