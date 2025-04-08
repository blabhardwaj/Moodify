import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import uuid
import time

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

# Mock mood classifier function (replace with your actual model)
def classify_mood(image):
    # This is where you'd implement your actual ML model
    # For demo purposes, we'll randomly select a mood
    moods = ["happy", "sad", "calm", "angry", "excited"]
    import random
    return random.choice(moods)

# Get song recommendation
def get_song_for_mood(mood):
    mood_to_search = {
        "happy": "happy hits feel good",
        "sad": "sad songs emotional",
        "calm": "peaceful relaxing",
        "angry": "intense rock metal",
        "excited": "upbeat party dance"
    }
    
    search_query = mood_to_search.get(mood.lower(), "popular")
    results = sp.search(q=search_query, type='track', limit=5)
    
    if results['tracks']['items']:
        # Pick a random track from results
        import random
        track = random.choice(results['tracks']['items'])
        return {
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'album': track['album']['name'],
            'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
            'preview_url': track['preview_url'],
            'external_url': track['external_urls']['spotify']
        }
    return None

# Process button
if st.button("✨ Analyze Mood"):
    if canvas_result.image_data is not None:
        # Show processing animation
        with st.spinner("Analyzing your sketch..."):
            # Save the image
            img = Image.fromarray((canvas_result.image_data).astype("uint8"))
            img_path = f"sketch_{uuid.uuid4().hex[:8]}.png"
            img.save(img_path)
            
            # Get the mood (replace with actual model prediction)
            mood = classify_mood(img)
            time.sleep(1)  # Simulate processing time
            
            # Get song recommendation
            song = get_song_for_mood(mood)
            
        # Display results
        st.success(f"Your sketch expresses a **{mood.upper()}** mood!")
        
        if song:
            st.markdown(f"""
            <div class="result-container">
                <h3>🎵 Your Mood-Matched Song</h3>
                <p><b>{song['name']}</b> by {song['artist']}</p>
                <p>From the album: {song['album']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display album art if available
            if song['image_url']:
                st.image(song['image_url'], width=200)
            
            # Play preview if available
            if song['preview_url']:
                st.audio(song['preview_url'])
            
            # Link to open in Spotify
            st.markdown(f"[Open in Spotify]({song['external_url']})")
        else:
            st.error("Sorry, couldn't find a matching song. Try another sketch!")
