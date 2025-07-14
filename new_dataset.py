from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
path = os.path.join(".", "tcc_ceds_music.csv")
data = pd.read_csv(path)
print(data.shape)
og = data.copy()
# print(data.head())
# print(data.describe())
# print(data.info())
transformed_data = data.drop(columns=[data.columns[0], 'lyrics', 'track_name'])
print(transformed_data.info())
encoder = LabelEncoder()
transformed_data['artist_name'] = encoder.fit_transform(
    transformed_data['artist_name'])
transformed_data['genre'] = encoder.fit_transform(transformed_data['genre'])
transformed_data['topic'] = encoder.fit_transform(transformed_data['topic'])
# Separate numerical features for standard scaling
numerical_features = transformed_data.drop(
    columns=['artist_name', 'genre', 'topic'])

# Initialize StandardScaler
standard_scaler = StandardScaler()

# Perform standard scaling on numerical features
scaled_numerical_features = standard_scaler.fit_transform(numerical_features)

# Replace original numerical features with scaled ones
transformed_data[numerical_features.columns] = scaled_numerical_features
# print(transformed_data.info())

# model = KMeans(n_clusters=20, init='k-means++')
# model.fit(transformed_data)
# clusters = model.predict(transformed_data)
# print(clusters)
# calinski_harabasz = calinski_harabasz_score(transformed_data, model.labels_)
# davies_bouldin = davies_bouldin_score(transformed_data, model.labels_)
# silhouette = silhouette_score(transformed_data, model.labels_)

# # Print evaluation metrics
# print(f"Silhouette Score: {silhouette}")
# print(f"Calinski-Harabasz Index: {calinski_harabasz}")
# print(f"Davies-Bouldin Index: {davies_bouldin}")


# clusters = model.predict(transformed_data)
# print(clusters)
# pca = PCA(n_components=10)
# music_data_transformed = pca.fit_transform(transformed_data)
# kmeans = KMeans(n_clusters=2, init='k-means++')
# clusters = kmeans.fit_predict(music_data_transformed)
# music_data_transformed = transformed_data
# clusters = model.predict(transformed_data)
# print(clusters)
# plt.figure(figsize=(8, 6))
# plt.scatter(music_data_transformed[:, 0],
#             music_data_transformed[:, 1],  c=clusters, cmap='viridis')
# plt.title('PCA with Cluster Plot')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.colorbar(label='Cluster')
# plt.show()


# using DBSCAN
# working
# model2 = DBSCAN(eps=7, min_samples=10)
# clusters = model2.fit_predict(transformed_data)
# for i in clusters:
#     print(i, end=' ')


# model2 = DBSCAN(eps=2, min_samples=3)
# clusters = model2.fit_predict(transformed_data)
# for i in clusters:
#     print(i, end=' ')
# silhouette = silhouette_score(transformed_data, clusters)
# calinski_harabasz = calinski_harabasz_score(transformed_data, clusters)
# davies_bouldin = davies_bouldin_score(transformed_data, clusters)

# print(f"Silhouette Score: {silhouette}")
# print(f"Calinski-Harabasz Index: {calinski_harabasz}")
# print(f"Davies-Bouldin Index: {davies_bouldin}")

# using PCA
# pca = PCA(n_components=2)
# transformed_data = pca.fit_transform(transformed_data)


# using tsne
# tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
# transformed_data = tsne.fit_transform(transformed_data)

tsne = TSNE(n_components=2, perplexity=30, n_iter=20000, random_state=42)
transformed_data = tsne.fit_transform(transformed_data)

# model = KMeans(n_clusters=9, init='k-means++')
# model.fit(transformed_data)
# clusters = model.predict(transformed_data)
# print(clusters)


# model2 = DBSCAN(eps=5, min_samples=10)
# clusters = model2.fit_predict(transformed_data)
# print(clusters)

# for i in range (len(og)):
#     if(clusters[i]==0):
#         print(i,og.iloc[i]["track_name"])

plt.figure(figsize=(8, 6))
# plt.scatter(transformed_data[:, 0], transformed_data[:, 1],
#             c=clusters, cmap='viridis')
plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
# plt.scatter(transformed_data[:, 0],
#             transformed_data[:, 1], transformed_data[:, 2])
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()
# Plot the clusters
