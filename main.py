from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import numpy as np
from sklearn.cluster import kmeans_plusplus, KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Loading dataset :
music_data = pd.read_csv("./data/data.csv")
data_after_2015 = filter(lambda x: x['year'] >= 2015, music_data)
print(data_after_2015)
print("Hi")
# Data Exploration


def data_exploration():
    print(f"Shape of data : {music_data.shape}")
    print("----------------------------------------------------------------------")
    print("Information about data : ")
    music_data.info()
    print("----------------------------------------------------------------------")
    print(f"No of null values : \n{music_data.isnull().sum()}")
    print("----------------------------------------------------------------------")
    print("Describing data : \n", music_data.describe())
    print("----------------------------------------------------------------------")


# data_exploration()


# Data Preprocessing :


def data_preprocessing():
    # remoiving duplicates
    music_data.drop_duplicates(subset=['name'], inplace=True)
    # removing release data
    music_data.drop(columns=['release_date', 'id'], inplace=True)
    # converting textual data into numeric
    columns = ['artists', 'name']
    encoder = LabelEncoder()
    for column in columns:
        music_data[column] = encoder.fit_transform(music_data[column])


data_preprocessing()
print(music_data.columns)
# Distribution plot
# for column in music_data.columns:
#     sns.displot(music_data[column])
#     plt.show()


# Training model

# Perform PCA
# pca = PCA(n_components=10)
# music_data_transformed = pca.fit_transform(music_data)
standard_scaler = StandardScaler()
music_data_transformed = standard_scaler.fit_transform(music_data)
# Perform clustering
# You can change the number of clusters as needed
# kmeans = KMeans(n_clusters=10)
# clusters = kmeans.fit_predict(music_data)
tsne = TSNE(n_components=2, perplexity=10, n_iter=1000, random_state=42)
transformed_data = tsne.fit_transform(music_data_transformed)

plt.figure(figsize=(8, 6))
plt.scatter(transformed_data[:, 0],
            transformed_data[:, 1])
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()

# Plot clusters
# plt.figure(figsize=(8, 6))
# plt.scatter(music_data_transformed[:, 0],
#             music_data_transformed[:, 1],  c=clusters, cmap='viridis')
# plt.title('PCA with Cluster Plot')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.colorbar(label='Cluster')
# plt.show()
# silhouette = silhouette_score(, kmeans.labels_)
# print(silhouette)
# plt.plot(X, Y)
# plt.show()
model = KMeans(n_clusters=10, max_iter=1000)
model.fit(music_data)
prediction = model.transform(music_data)
# print(min(prediction[0]))
print(model.inertia_)
print(prediction[0])

all_song_cluster = model.predict(music_data)
print(all_song_cluster)

# Getting results
# cluster_song = music_data[all_song_cluster == prediction[0]]

# Calculate evaluation metrics
calinski_harabasz = calinski_harabasz_score(music_data, model.labels_)
davies_bouldin = davies_bouldin_score(music_data, model.labels_)
silhouette = silhouette_score(music_data, model.labels_)

# # Print evaluation metrics
# print(f"Silhouette Score: {silhouette}")
print(f"Calinski-Harabasz Index: {calinski_harabasz}")
print(f"Davies-Bouldin Index: {davies_bouldin}")
