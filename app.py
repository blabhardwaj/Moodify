import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
from ml_pipeline import MoodClassifierPipeline
from gemini_helper import init_gemini, generate_mood_phrase
import streamlit.components.v1 as components
from colorthief import ColorThief
import requests
from io import BytesIO
import random
from streamlit_extras.stylable_container import stylable_container
import re

st.set_page_config(
    page_title="Moodify My Sketch",
    page_icon="🎨",
    layout="wide"
)

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

col1, col2 = st.columns([3, 1], gap="large")

with col1:
    preset_colors = [
        "#000000", "#FF0000", "#00FF00", "#0000FF",
        "#FFFF00", "#FFA500", "#800080", "#00FFFF"
    ]

    if "stroke_color" not in st.session_state:
        st.session_state.stroke_color = "#000000"

    color_cols = st.columns(len(preset_colors) + 1, gap="small")

    for i, color in enumerate(preset_colors):
        with color_cols[i]:
            with stylable_container(
                f"color_btn_{i}",
                css_styles=f"""
                button {{
                    background-color: {color} !important;
                    border: 2px solid #888 !important;
                    width: 40px !important;
                    height: 40px !important;
                    border-radius: 8px !important;
                    padding: 0 !important;
                    margin: 0 !important;
                }}
                """,
            ):
                if st.button(" ", key=f"color_btn_{i}"):
                    st.session_state.stroke_color = color

    # Multi-color picker
    with color_cols[-1]:
        picked = st.color_picker(
            " ", st.session_state.stroke_color, label_visibility="collapsed", key="stroke_picker"
        )
        st.markdown(
            f"""
            <style>
            [data-testid="stColorPicker"] > div {{
                width: 45px !important;
                height: 45px !important;
                border-radius: 7px !important;
                border: 3px solid #888 !important;
                margin: 0 !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        if picked != st.session_state.stroke_color:
            st.session_state.stroke_color = picked

    stroke_color = st.session_state.stroke_color

    # Stroke width slider
    stroke_width = st.slider("Stroke width", 1, 25, 3, label_visibility="collapsed")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.8)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#FFFFFF",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
    )

with col2:
    st.markdown('<div style="font-size:1.2rem; font-weight:600; margin-bottom:8px;">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ol style="color:#444; font-size:1.05rem; margin-bottom:18px;">
            <li>Draw your mood</li>
            <li>Click <b>Analyze Mood</b></li>
            <li>Get a matching song!</li>
        </ol>
        <div style="font-size:1.2rem; font-weight:600; margin-bottom:8px;">Mood Options</div>
        <ul style="color:#444; font-size:1.05rem; list-style:none; padding-left:0;">
            <li>😊 Happy</li>
            <li>😢 Sad</li>
            <li>😌 Calm</li>
            <li>😠 Angry</li>
            <li>🤩 Excited</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

def is_canvas_blank(image_array):
    if image_array is None:
        return True
    image_uint8 = (image_array[:, :, :3]).astype(np.uint8)
    return np.all(image_uint8 == 255)

def is_hindi_or_punjabi(text):
    # Naive approach: check for Hindi or Punjabi Unicode blocks
    # Hindi and Punjabi characters are in Devanagari and Gurmukhi scripts
    devanagari_pattern = r'[\u0900-\u097F]'
    gurmukhi_pattern = r'[\u0A00-\u0A7F]'
    return bool(re.search(devanagari_pattern, text)) or bool(re.search(gurmukhi_pattern, text))

def is_desi_artist(artist_id):
    DESI_GENRES = {
        "bollywood", "punjabi", "desi", "indian pop", "hindustani", "telugu",
        "tamil", "bhangra", "indian indie", "desi hip hop", "dhh", "hindi",
        "ghazal", "qawwali", "sufi", "indian classical", "kannada", "marathi",
        "malayalam", "bhojpuri", "rajasthani", "gujarati", "odia", "folk",
        "indian fusion", "carnatic", "hindustani classical", "rabindra sangeet"
    }
    try:
        artist = sp.artist(artist_id)
        genres = artist.get("genres", [])
        for genre in genres:
            if any(desi_gen in genre.lower() for desi_gen in DESI_GENRES):
                return True
    except Exception as e:
        print(f"Error fetching artist genres: {e}")
    return False

def get_song_for_mood(search_term):
    retries = 3
    delay = 1.5
    search_query = search_term.strip().lower()

    for attempt in range(retries):
        try:
            results = sp.search(q=search_query, type='track', limit=20)
            tracks = results.get('tracks', {}).get('items', [])
            popular_tracks = [track for track in tracks if track.get('popularity', 0) >= 40]

            desi_tracks = []
            for track in popular_tracks:
                artist_id = track['artists'][0]['id']
                artist_name = track['artists'][0]['name']
                if is_hindi_or_punjabi(track['name']) or is_hindi_or_punjabi(artist_name) or is_desi_artist(artist_id):
                    desi_tracks.append(track)

            if desi_tracks:
                track = random.choice(desi_tracks)
            elif popular_tracks:
                track = random.choice(popular_tracks)
            else:
                print(f"Attempt {attempt + 1} found no popular tracks. Retrying...")
                time.sleep(delay)
                continue

            return {
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                'external_url': track['external_urls']['spotify'],
                'uri': track['uri'],
                'popularity': track.get('popularity', 0)
            }

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
            display_phrase, spotify_query, moodstat = generate_mood_phrase(gemini, img, ml_mood)
            
            #st.code(f"ML Mood: {ml_mood}\nSpotify Search Query: {spotify_query}\nDisplay Phrase: {display_phrase}\nmoodstat: {moodstat}", language='text')

            time.sleep(1)
            
            song = get_song_for_mood(f"{spotify_query}")

            if not song:
                song = get_song_for_mood(f"{display_phrase}")

            if not song:
                song = get_song_for_mood(f"{ml_mood} music")
            

        st.success(f"🖌️ Your sketch expresses a **{display_phrase}** vibe.")

        if song:
            st.markdown("## 🎵 A song that matches your sketch's vibe")

            #embed container
            track_id = song['uri'].split(':')[-1]
            embed_url = f"https://open.spotify.com/embed/track/{track_id}"

            dominant_color = get_dominant_color(song['image_url']) if song['image_url'] else "#FFFFFF"
            text_color = get_text_color(dominant_color)
            secondary_text_color = "#222" if text_color == "#000" else "#e0e0e0"

            st.markdown(
                f"""
                <div style="
                    background-color: {dominant_color};
                    padding: 18px;
                    border-radius: 12px;
                    margin-bottom: 20px;
                    max-width: 90%;
                    margin-left: auto;
                    margin-right: auto;
                    ">
                    <h4 style="margin-bottom: 5px; color: {text_color}; text-align:center;">{song['name']} <span style="font-weight: normal;">by {song['artist']}</span></h4>
                    <p style="margin-top: -10px; color: {secondary_text_color}; text-align:center;">Album: <em>{song['album']}</em></p>
                    <div style="display: flex; justify-content: center;">
                        <div style="border-radius: 10px; overflow: hidden; margin-top: 10px; width: 100%;">
                            <iframe src="{embed_url}" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("No good song match found. Try a different sketch or color!")