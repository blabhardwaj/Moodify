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

def get_song_for_mood(search_term, retries=3, delay=1.5):
    search_query = search_term.strip().lower()

    for attempt in range(retries):
        try:
            results = sp.search(q=search_query, type='track', limit=10)
            tracks = results.get('tracks', {}).get('items', [])
            for track in tracks:
                if track.get('preview_url'):
                    return {
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'album': track['album']['name'],
                        'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                        'preview_url': track['preview_url'],
                        'external_url': track['external_urls']['spotify'],
                        'uri': track['uri']
                    }
            return None  # No track with preview found

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)

    print("Spotify search failed after retries.")
    return None


if st.button("✨ Analyze Mood"):
    if canvas_result.image_data is None or is_canvas_blank(canvas_result.image_data):
        st.error("Please draw something first!")
    else:
        with st.spinner("Analyzing your sketch..."):
            img = Image.fromarray((canvas_result.image_data).astype("uint8"))
            img.save(f"sketch_{uuid.uuid4().hex}.png")

            ml_mood = classifier.classify_mood(canvas_result.image_data)
            display_phrase, spotify_query = generate_mood_phrase(gemini, img, ml_mood)
            
            st.code(f"ML Mood: {ml_mood}\nSpotify Search Query: {spotify_query}\nDisplay Phrase: {display_phrase}", language='text')

            time.sleep(1)
            songs = get_song_for_mood(spotify_query)

            if not songs:
                songs = get_song_for_mood(f"{ml_mood} mood music")

        st.success(f"🖌️ Your sketch expresses a **{display_phrase}** vibe.")

       

        if songs:
            st.markdown("## 🎵 Songs that match your sketch's vibe")

            for idx, song in enumerate(songs, 1):
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                        <h4 style="margin-bottom: 5px;">{idx}. {song['name']} <span style="font-weight: normal;">by {song['artist']}</span></h4>
                        <p style="margin-top: -10px; color: #666;">Album: <em>{song['album']}</em></p>
                    """,
                    unsafe_allow_html=True
                )

                cols = st.columns([1, 3])
                with cols[0]:
                    if song['image_url']:
                        st.image(song['image_url'], width=90)
                with cols[1]:
                    st.markdown(
                        f"""
                        <iframe src="https://open.spotify.com/embed/track/{song['uri'].split(':')[-1]}" 
                        width="100%" height="80" frameborder="0" 
                        allowtransparency="true" allow="encrypted-media"
                        style="border-radius: 10px;">
                        </iframe>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)


        else:
            st.error("No good song match found. Try a different sketch or color!")
