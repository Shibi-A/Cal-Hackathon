import spotipy
import os
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
# Set up Spotify credentials (from Spotify Developer Dashboard)
#client_id = os.getenv("spotify_client_id")  # Replace with your Spotify client ID
#client_secret = os.getenv("spotify_client_secret")  # Replace with your Spotify client secret
load_dotenv()
client_secret = os.getenv("spotify_client_secret")
client_id = os.getenv("spotify_client_id")
# Authentication - without user login
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to get top 100 tracks
def get_top_100_songs():
    # Fetch the Global Top 50 playlist first
    top_50_global = sp.playlist_tracks('37i9dQZEVXbMDoHDwVN2tF')  # Global Top 50 Playlist ID
    top_50_usa = sp.playlist_tracks('37i9dQZEVXbLRQDuF5jeBp')  # USA Top 50 Playlist ID

    top_tracks = []

    # Extract tracks from both playlists and combine them
    for item in top_50_global['items'] + top_50_usa['items']:
        track = item['track']
        track_info = {
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'popularity': track['popularity'],
            'url': track['external_urls']['spotify'],
        }
        if track_info not in top_tracks:  # Avoid duplicates
            top_tracks.append(track_info)
    
    return top_tracks[:50]  # Return only the top 100 unique tracks
"""
# Fetch and display the top 100 tracks
top_100_songs = get_top_100_songs()

print("Top 100 Spotify Songs:")
for idx, song in enumerate(top_100_songs, 1):
    print(f"{idx}. {song['name']} by {song['artist']} (Popularity: {song['popularity']}) - {song['url']}")
"""