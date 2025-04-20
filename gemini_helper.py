# gemini_helper.py

from google import genai
from PIL import Image
import io

def init_gemini(api_key):
    client = genai.Client(api_key=api_key)
    return client.models

def generate_mood_phrase(models, image: Image.Image, ml_mood: str):
    """Send image + ML mood to Gemini for natural language mood interpretation."""
    prompt = (
    f"This is a drawing labeled with the emotion '{ml_mood}'. "
    "Describe the emotional atmosphere using a few musical or emotional keywords (max 6 words). "
    "Use words that could describe a song’s vibe—like genres, emotions, tempo, or energy."
)


    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    
    response = models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            genai.types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        ]
    )
    return response.text.strip()
