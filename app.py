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
import streamlit.components.v1 as components
from colorthief import ColorThief
import requests
from io import BytesIO
# Set page config
st.set_page_config(
    page_title="Moodify My Sketch",
    page_icon="🎨",
    layout="wide"
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

@st.cache_resource
def load_pipeline():
    return MoodClassifierPipeline()

classifier = load_pipeline()

@st.cache_resource
def load_gemini():
    return init_gemini(st.secrets.get("GEMINI_API_KEY"))

gemini = load_gemini()

@st.cache_resource
def get_spotify_client():
    try:
        client_id = st.secrets["SPOTIFY_CLIENT_ID"]
        client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
    except KeyError:
        raise RuntimeError("Spotify credentials missing in Streamlit secrets.")

    client_credentials_manager = SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    )
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)


sp = get_spotify_client()

# Drawing canvas
col1, col2 = st.columns([3, 1])

with col1:
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
    st.write("### Mood Options")
    st.write("- Happy 😊\n- Sad 😢\n- Calm 😌\n- Angry 😠\n- Excited 🤩")

def is_canvas_blank(image_array):
    if image_array is None:
        return True
    image_uint8 = (image_array[:, :, :3]).astype(np.uint8)
    return np.all(image_uint8 == 255)

import time

def get_song_for_mood(search_term):
    retries = 3
    delay = 1.5
    search_query = search_term.strip().lower()

    for attempt in range(retries):
        try:
            results = sp.search(q=search_query, type='track', limit=10)
            tracks = results.get('tracks', {}).get('items', [])
            # Filter tracks by popularity (e.g., >= 50)
            popular_tracks = [track for track in tracks if track.get('popularity', 0) >= 50]
            if popular_tracks:
                track = popular_tracks[0]
                return {
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'external_url': track['external_urls']['spotify'],
                    'uri': track['uri'],
                    'popularity': track.get('popularity', 0)
                }
            print(f"Attempt {attempt + 1} found no popular tracks. Retrying...")
            time.sleep(delay)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)

    print("Spotify search failed after retries.")
    return None

def get_dominant_color(image_url):
    response = requests.get(image_url)
    color_thief = ColorThief(BytesIO(response.content))
    dominant_color = color_thief.get_color(quality=1)
    # Convert (R, G, B) to hex
    return '#%02x%02x%02x' % dominant_color

def get_text_color(bg_hex):
    # Convert hex to RGB
    bg_hex = bg_hex.lstrip('#')
    r, g, b = tuple(int(bg_hex[i:i+2], 16) for i in (0, 2, 4))
    # Calculate luminance
    luminance = (0.299*r + 0.587*g + 0.114*b)/255
    return "#000" if luminance > 0.6 else "#fff"

if st.button("✨ Analyze Mood"):
    if canvas_result.image_data is None or is_canvas_blank(canvas_result.image_data):
        st.error("Please draw something first!")
    else:
        with st.spinner("Analyzing your sketch..."):
            img = Image.fromarray((canvas_result.image_data).astype("uint8"))

            ml_mood = classifier.classify_mood(canvas_result.image_data)
            display_phrase, spotify_query = generate_mood_phrase(gemini, img, ml_mood)
            
            #st.code(f"ML Mood: {ml_mood}\nSpotify Search Query: {spotify_query}\nDisplay Phrase: {display_phrase}", language='text')

            time.sleep(1)
            
            song = get_song_for_mood(f"{spotify_query}")

            if not song:
                song = get_song_for_mood(f"{display_phrase}")

            if not song:
                song = get_song_for_mood(f"{ml_mood} music")
            

        st.success(f"🖌️ Your sketch expresses a **{display_phrase}** vibe.")

        if song:
            st.markdown("## 🎵 A song that matches your sketch's vibe")

            # Card-like container with details and embed player (no image)
            track_id = song['uri'].split(':')[-1]
            embed_url = f"https://open.spotify.com/embed/track/{track_id}"

            dominant_color = get_dominant_color(song['image_url']) if song['image_url'] else "#FFFFFF"
            text_color = get_text_color(dominant_color)
            secondary_text_color = "#222" if text_color == "#000" else "#e0e0e0"

            st.markdown(
                f"""
                <div style="background-color: {dominant_color}; padding: 18px; border-radius: 12px; margin-bottom: 20px;">
                    <h4 style="margin-bottom: 5px; color: {text_color};"> {song['name']} <span style="font-weight: normal;">by {song['artist']}</span></h4>
                    <p style="margin-top: -10px; color: {secondary_text_color};">Album: <em>{song['album']}</em></p>
                    <div style="position: relative; margin-top: 18px;">
                        <div style="position: relative; z-index: 1; border-radius: 10px; overflow: hidden;">
                            <iframe src="{embed_url}" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("No good song match found. Try a different sketch!")