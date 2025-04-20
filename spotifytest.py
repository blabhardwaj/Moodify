import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import streamlit.components.v1 as components

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

'''def get_song_for_mood(search_term):
    """
    Search Spotify using the mood search term.
    Return a list of up to 5 high-quality, previewable tracks (popularity ≥ 40).
    """
    search_query = search_term.strip().lower()
    try:
        results = sp.search(q=search_query, type='track', limit=20)
        tracks = results.get('tracks', {}).get('items', [])
        filtered = []

        for track in tracks:
            popularity = track.get('popularity', 0)
            if track.get('preview_url') and popularity >= 40:
                filtered.append({
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'preview_url': track['preview_url'],
                    'external_url': track['external_urls']['spotify'],
                    'uri': track['uri'],
                    'popularity': popularity
                })

            if len(filtered) == 5:
                break

        return filtered

    except Exception as e:
        print(f"Spotify error: {e}")
        return []

'''

search_term = "mellow acoustic"
results = sp.search(q=search_term, type="track", limit=5)
for idx, track in enumerate(results['tracks']['items']):
    track_id = track['id']
    track_name = track['name']
    artist_name = track['artists'][0]['name']
    st.write(f"{idx+1}. {track_name} by {artist_name}")
    embed_url = f"https://open.spotify.com/embed/track/{track_id}"
    components.iframe(embed_url, height=80, width=300)