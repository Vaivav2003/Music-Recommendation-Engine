import joblib
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
# Assuming database have been clustered and their cluster number have been stored in separate column NOW RECOMMEND SONGS
input_song = pd.read_csv(
    "D:\\College Study\\Practicum\\test_data.csv")
track_name = []
for i in range(len(input_song)):
    track_name.append(input_song["track_name"].iloc[i])
input_song = input_song.drop(
    columns=[input_song.columns[0], 'lyrics', 'track_name'])

data_with_cluster = pd.read_csv(
    "D:\\College Study\\Practicum\\data_with_cluster.csv")
loaded_model = joblib.load('model_filename.pkl')


def recommend_songs(song):

    encoder = LabelEncoder()
    song['artist_name'] = encoder.fit_transform(
        song['artist_name'])
    song['genre'] = encoder.fit_transform(
        song['genre'])
    song['topic'] = encoder.fit_transform(
        song['topic'])

    # Standardizing
    standard_scaler = StandardScaler()

    # Separate numerical features for standard scaling
    numerical_features = song.drop(
        columns=['artist_name', 'genre', 'topic'])

    # Perform standard scaling on numerical features
    scaled_numerical_features = standard_scaler.fit_transform(
        numerical_features)

    # Replace original numerical features with scaled ones
    song[numerical_features.columns] = scaled_numerical_features

    tsne = TSNE(n_components=2, perplexity=1, n_iter=250, random_state=42)
    song = tsne.fit_transform(song)
    cluster_number = loaded_model.fit_predict(song)
    similar_songs = []
    for i in range(len(data_with_cluster)):
        similar = data_with_cluster.iloc[i]
        if (similar["cluster number"] == cluster_number.any()):
            similar_songs.append(similar["track_name"])
        if (len(similar_songs) == 10):
            break
    return similar_songs


print("Similar Songs : ")
recommeded = recommend_songs(input_song)
for song in recommeded:
    print(song)
