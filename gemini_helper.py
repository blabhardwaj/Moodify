from google import genai
from PIL import Image
import io

def init_gemini(api_key):
    client = genai.Client(api_key=api_key)
    return client.models

def generate_mood_phrase(models, image: Image.Image, ml_mood: str):
    prompt = f"""
    Given the mood "{ml_mood}" detected from a sketch, analyze the sketch, look for shapes and what they mean, look for text and generate:
    1. A poetic and emotional phrase to describe the mood (for user display).
    2. A simplified Spotify music search query that captures the same emotion but uses real-world language to return good music results (e.g., 'sad indie', 'calm lofi', 'chill pop', etc.).
    3. Recheck the mood prediction and return 'yes' or 'no' if the mood prediction was right + correct mood. 

    Keep it natural, 4-6 words, and not overly abstract.

    Output in this exact format:
    DISPLAY_PHRASE: <display phrase>
    SPOTIFY_QUERY: <spotify search query>
    moodstat: <was the mood prediction right? - yes or no + correct mood>
    """

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

    display_phrase = ""
    spotify_query = ""
    moodstat = ""

    try:
        for line in response.text.strip().splitlines():
            if line.startswith("DISPLAY_PHRASE:"):
                display_phrase = line.replace("DISPLAY_PHRASE:", "").strip()
            elif line.startswith("SPOTIFY_QUERY:"):
                spotify_query = line.replace("SPOTIFY_QUERY:", "").strip()
            elif line.startswith("moodstat:"):
                moodstat = line.replace("moodstat:", "").strip()
    except Exception as e:
        print("Error parsing Gemini response:", e)
        print("Raw Gemini output:", response.text)

    # Fallbacks
    if not display_phrase:
        display_phrase = f"A sketch that feels {ml_mood}"
    if not spotify_query:
        spotify_query = f"{ml_mood} mood music"

    return (display_phrase, spotify_query, moodstat)
