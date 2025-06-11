import streamlit as st
import pickle
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np

CLIENT_ID = "Client_id" 
CLIENT_SECRET = "Client_Secret" 

@st.cache_resource
def load_all_data_and_models():
    data = {}

    # Attempt to load text-based data and models
    try:
        with open('nn_model_text.pkl', 'rb') as f:
            data['nn_model_text'] = pickle.load(f)
        with open('tfidf_matrix.pkl', 'rb') as f:
            data['tfidf_matrix'] = pickle.load(f)
        with open('df_text.pkl', 'rb') as f:
            data['df_text'] = pickle.load(f)
        data['text_based_loaded'] = True
        print("Loaded text-based models and data (cached).")
    except FileNotFoundError:
        data['text_based_loaded'] = False
        print("Text-based models not found.")

    # Attempt to load feature-based data and models
    try:
        with open('nn_model_features.pkl', 'rb') as f:
            data['nn_model_features'] = pickle.load(f)
        with open('scaled_features.pkl', 'rb') as f:
            data['scaled_features'] = pickle.load(f)
        with open('df_features.pkl', 'rb') as f:
            data['df_features'] = pickle.load(f)
        data['feature_based_loaded'] = True
        print("Loaded feature-based models and data (cached).")
    except FileNotFoundError:
        data['feature_based_loaded'] = False
        print("Feature-based models not found.")

    if not data['text_based_loaded'] and not data['feature_based_loaded']:
        st.error("Could not load any recommendation models. Please train and save them.")
        st.stop()

    return data

data = load_all_data_and_models()

@st.cache_resource
def init_spotify():
    try:
        client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        print("Successfully connected to Spotify API (cached).")
        return sp, True
    except Exception as e:
        st.warning(f"Could not connect to Spotify API: {e}")
        st.warning("Album covers will not be displayed.")
        return None, False

sp, spotify_connected = init_spotify()

def get_song_album_cover_url(song_name, artist_name):
    if not spotify_connected:
        return "https://i.postimg.cc/0QNxYz4V/social.png" # Default image

    try:
        search_query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=search_query, type="track")
        if results and results["tracks"]["items"]:
            track = results["tracks"]["items"][0]
            if track["album"]["images"]:
                return track["album"]["images"][0]["url"]
        return "https://i.postimg.cc/0QNxYz4V/social.png"  # Default image if no results or no images
    except Exception as e:
        print(f"Error fetching Spotify data for {song_name} by {artist_name}: {e}")
        return "https://i.postimg.cc/0QNxYz4V/social.png" # Default image on error

# Recommendation function
def recommender(song_title):
    recommended_music_names = []
    recommended_music_posters = []
    found_in_text = False
    found_in_feature = False

    # Try text-based recommendation
    if data.get('text_based_loaded'):
        try:
            df_text = data['df_text']
            nn_model_text = data['nn_model_text']
            tfidf_matrix = data['tfidf_matrix']

            song_index = df_text[df_text['song'] == song_title].index[0]
            song_vector = tfidf_matrix[song_index]
            distances, indices = nn_model_text.kneighbors(song_vector)

            recommended_song_indices = indices.flatten()[1:]
            for i in recommended_song_indices:
                recommended_music_names.append(df_text.iloc[i]['song'])
                recommended_music_posters.append(get_song_album_cover_url(df_text.iloc[i]['song'], df_text.iloc[i]['artist']))
            found_in_text = True
            st.write(f"Recommendations for '{song_title}' (Text-based):") # Indicate which method is used

        except IndexError:
            print(f"'{song_title}' not found in text-based dataset.")

    # If not found in text-based or text-based not loaded, try feature-based
    if not found_in_text and data.get('feature_based_loaded'):
        try:
            df_features = data['df_features']
            nn_model_features = data['nn_model_features']
            scaled_features = data['scaled_features']

            song_index = df_features[df_features['track_name'] == song_title].index[0]
            song_features = scaled_features[song_index].reshape(1, -1)
            distances, indices = nn_model_features.kneighbors(song_features)

            recommended_song_indices = indices.flatten()[1:]
            recommended_music_names = [] # Clear previous recommendations if text-based failed
            recommended_music_posters = [] # Clear previous posters

            for i in recommended_song_indices:
                recommended_music_names.append(df_features.iloc[i]['track_name'])
                recommended_music_posters.append(get_song_album_cover_url(df_features.iloc[i]['track_name'], df_features.iloc[i]['artist(s)_name']))
            found_in_feature = True
            st.write(f"Recommendations for '{song_title}' (Feature-based):") # Indicate which method is used

        except IndexError:
            print(f"'{song_title}' not found in feature-based dataset.")

    if not found_in_text and not found_in_feature:
        st.warning(f"'{song_title}' not found in either dataset.")

    return recommended_music_names, recommended_music_posters

# Streamlit App Layout
st.header('Music Recommender System')

# Combine song lists from both datasets if loaded
music_list = []
if data.get('text_based_loaded'):
    music_list.extend(data['df_text']['song'].tolist())
if data.get('feature_based_loaded'):
    music_list.extend(data['df_features']['track_name'].tolist())

# Remove duplicates and sort
music_list = sorted(list(set(music_list)))

selected_song = st.selectbox(
    "Type or select a song from the dropdown",
    music_list
)

if st.button('Show Recommendation'):
    with st.spinner('Getting recommendations...'):
        recommended_music_names, recommended_music_posters = recommender(selected_song)

    if recommended_music_names:
        num_recommendations = len(recommended_music_names)
        # Use min(num_recommendations, 5) or a fixed number of columns if you prefer
        cols = st.columns(min(num_recommendations, 5)) # Create up to 5 columns

        for i in range(min(num_recommendations, 5)): # Iterate up to 5 recommendations
            with cols[i]:
                st.text(recommended_music_names[i])
                st.image(recommended_music_posters[i])
    else:
        # The recommender function already prints a warning if not found
        pass
