import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import uuid
import time
from ml_pipeline import MoodClassifierPipeline
from gemini_helper import init_gemini, generate_mood_phrase


# Set page config
st.set_page_config(
    page_title="Moodify My Sketch",
    page_icon="🎨",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background-color: #1DB954;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .result-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎨 Moodify My Sketch")
st.write("Draw something that shows your mood, and we'll match a song to it!")


# Cache the loading of the MoodClassifierPipeline to avoid reloading it multiple times
@st.cache_resource
def load_pipeline():
    return MoodClassifierPipeline()

# Initialize the mood classifier pipeline
classifier = load_pipeline()

# Cache the initialization of the Gemini API helper to avoid redundant API key setups
@st.cache_resource
def load_gemini():
    return init_gemini(st.secrets.get("GEMINI_API_KEY", "your_gemini_key"))

# Initialize the Gemini API helper
gemini = load_gemini()


# Initialize Spotify client
@st.cache_resource
def get_spotify_client():
    client_id = st.secrets.get("SPOTIFY_CLIENT_ID", "your_client_id")
    client_secret = st.secrets.get("SPOTIFY_CLIENT_SECRET", "your_client_secret")
    client_credentials_manager = SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    )
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

sp = get_spotify_client()

# Drawing canvas setup
col1, col2 = st.columns([3, 1])

with col1:
    # Canvas for drawing
    stroke_color = st.color_picker("Stroke color", "#000000")
    stroke_width = st.slider("Stroke width", 1, 25, 3)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.8)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#FFFFFF",
        height=350,
        width=350,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
    )

with col2:
    st.write("### How it works")
    st.write("1. Draw your mood")
    st.write("2. Click 'Analyze Mood'")
    st.write("3. Get a matching song!")
    
    # Mood mapping
    st.write("### Mood Options")
    st.write("- Happy 😊")
    st.write("- Sad 😢")
    st.write("- Calm 😌")
    st.write("- Angry 😠")
    st.write("- Excited 🤩")

#checks if the canvas is blank
def is_canvas_blank(image_array):
    # Check if all pixels are white (255, 255, 255)
    if image_array is None:
        return True
    image_uint8 = (image_array[:, :, :3]).astype(np.uint8)  # Remove alpha if present
    return np.all(image_uint8 == 255)  # All white


# Mock mood classifier function (replace with your actual model)
def classify_mood(image):
    # This is where you'd implement your actual ML model
    # For demo purposes, we'll randomly select a mood
    moods = ["happy", "sad", "calm", "angry", "excited"]
    import random
    return random.choice(moods)

# Get song recommendation
def get_song_for_mood(mood_phrase):
    """
    Search Spotify using the Gemini-generated mood phrase directly.
    Return a previewable track if available.
    """
    # Clean up overly poetic lines (just in case)
    search_query = mood_phrase.strip().lower()
    
    try:
        results = sp.search(q=search_query, type='track', limit=10)
        tracks = results.get('tracks', {}).get('items', [])

        # Pick the first previewable track
        for track in tracks:
            if track['preview_url']:
                return {
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'preview_url': track['preview_url'],
                    'external_url': track['external_urls']['spotify'],
                    'uri': track['uri']  # Needed for embedded player
                }
    except Exception as e:
        print(f"Spotify error: {e}")

    return None


# Process button
# Process button
if st.button("✨ Analyze Mood"):
    if canvas_result.image_data is None or is_canvas_blank(canvas_result.image_data):
        st.error("Please draw something first!")

    else:    
        with st.spinner("Analyzing your sketch..."):
            # Convert image
            img = Image.fromarray((canvas_result.image_data).astype("uint8"))

            # Classify mood via ML
            ml_mood = classifier.classify_mood(canvas_result.image_data)

            # Use Gemini for refined emotional text
            mood_phrase = generate_mood_phrase(gemini, img, ml_mood)

            st.code(f"ML Mood: {ml_mood}\nGemini Phrase: {mood_phrase}", language='text')

            search_term = mood_phrase.split(",")[0].strip() + " music"
            
            time.sleep(1)

            # Get song based on mood phrase
            song = get_song_for_mood(search_term)

            if not song:
                song = get_song_for_mood(f"{ml_mood} mood music")
        
        st.success(f"Your sketch expresses a mood: **{mood_phrase}**")
        
        if song:
            st.markdown(f"""
            <div class="result-container">
                <h3>🎵 Your Mood-Matched Song</h3>
                <p><b>{song['name']}</b> by {song['artist']}</p>
                <p>From the album: {song['album']}</p>
            </div>
            """, unsafe_allow_html=True)

            if song['image_url']:
                st.image(song['image_url'], width=200)
            if song['preview_url']:
                st.audio(song['preview_url'])
            st.markdown(f"""
                        <iframe src="https://open.spotify.com/embed/track/{song['uri'].split(':')[-1]}" 
                        width="100%" height="80" frameborder="0" 
                        allowtransparency="true" allow="encrypted-media"></iframe>
                        """, unsafe_allow_html=True)

        else:
            st.error("Couldn't find a song. Try a different sketch.")
