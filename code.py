# Loading data

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

path = os.path.join(".", "tcc_ceds_music.csv")
dataset = pd.read_csv(path)
dataset.head()
# DATA EXPLORATION


def data_exploration():
    print(f"Shape of data : {dataset.shape}")
    print("----------------------------------------------------------------------------------")
    print(f"Data information : ")
    dataset.info()
    print("----------------------------------------------------------------------------------")
    print(f"Number of null values :\n {dataset.isnull().sum()}")
    print("----------------------------------------------------------------------------------")
    print(f"Data Description : \n{dataset.describe()}")


data_exploration()

# unique values in dataset


def unique_values(df):
    unique_values = []
    col_name = []
    null_values = []
    for col in df.columns:
        unique_values.append(df[col].nunique())
        col_name.append(col)
        null_values.append(df[col].isna().any())
    df_dict = {
        'Features': col_name,
        'Unique values': unique_values,
        'Contains Null': null_values
    }
    return pd.DataFrame(df_dict, columns=['Features', 'Unique values', 'Contains Null'])


unique_values(dataset)
# Genre wise distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 8))
genres = dataset['genre'].value_counts()
label = dataset['topic'].value_counts().values
highlight_start = 0
highlight_end = 1

# Plotting the line plot segments with different colors
# Segment 1 (before yellow section)
axes[0].plot(genres.index[:highlight_start],
             genres.values[:highlight_start], color='gray')

# Segment 2 (yellow section)
axes[0].plot(genres.index[highlight_start:highlight_end + 1],
             genres.values[highlight_start:highlight_end + 1], color='darkblue')

# Segment 3 (after yellow section)
axes[0].plot(genres.index[highlight_end:],
             genres.values[highlight_end:], color='gray')

# pie plot


def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%".format(pct)


wp = {'linewidth': 1, 'edgecolor': "green"}
patches, texts, autotexts = axes[1].pie(genres.values, labels=genres.index.values,
                                        autopct=lambda pct: func(
                                            pct, genres.values),
                                        explode=(0.1, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0),
                                        shadow=True,
                                        wedgeprops=wp,
                                        labeldistance=1.2)
# Example: Set the fontsize for better visibility

axes[0].set_ylabel('Total songs')
axes[0].set_xlabel('Genres')
axes[0].set_title('Genre distribution')
plt.tight_layout()

# add text
for x, y in zip(genres.index, genres.values):
    axes[0].text(x, y, f'{y}', ha='left', va='bottom',
                 fontsize='xx-small', stretch='expanded', fontweight='semibold')

for text in texts:
    text.set_fontweight('bold')
    text.set_horizontalalignment('center')

# add text
axes[0].text('jazz', 6890, 'Pop and Country\n genres together\n account for 44%\n of the dataset.',
             ha='left', va='center', fontsize=12, color='darkblue')
plt.show()


# Assuming you have a DataFrame 'df_songs' and want to plot the count of each genre
plt.figure(figsize=(8, 6))
g = sns.countplot(data=dataset, x='topic',
                  order=dataset['topic'].value_counts().index)


# Adding labels and title
plt.xlabel('Song themes')
plt.ylabel('Total songs')
plt.title('Genre in Numbers', loc='left')

# mean value
mean = dataset['topic'].value_counts().mean()

# adding mean horizontal line
g.axhline(y=mean, alpha=0.4, c='gray', linestyle='--', label='mean')


# ignoring color for lower last four bars
for i, bar in enumerate(g.patches):
    if i >= 4:  # Excluding top four bars
        bar.set_color('gray')
        bar.set_alpha(0.3)
    if i < 4:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(int(bar.get_height())), ha='center', va='bottom')


# Rotating x-axis labels for better readability (optional)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Extracting only numerical features for correlation
X_numeric = dataset.iloc[:, 5:30]
X_numeric = X_numeric.drop(['topic', 'lyrics'], axis=1)

# # compute the correlation matrix
numeric_corr = X_numeric.corr()

# # Generatae a mask for upper triangle
mask = np.triu(np.ones_like(numeric_corr, dtype=bool))

# increase figisze
figure, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(
    data=numeric_corr, mask=mask, annot=True, fmt='0.1f')
# REMOVING UNNECESSARY FEATURES ( lyrics , track_name )
# Removed 'Unnamed','lyrics','track_name'
transformed_data = dataset.drop(
    columns=[dataset.columns[0], 'lyrics', 'track_name'])
# TRANSFORMING TEXTUAL DATA (artist_name,genre,topic) INTO NUMERIC DATA
# USING LABEL ENCODER

encoder = LabelEncoder()
transformed_data['artist_name'] = encoder.fit_transform(
    transformed_data['artist_name'])
transformed_data['genre'] = encoder.fit_transform(transformed_data['genre'])
transformed_data['topic'] = encoder.fit_transform(transformed_data['topic'])

# USING StandardScaler
standard_scaler = StandardScaler()

# Separate numerical features for standard scaling
numerical_features = transformed_data.drop(
    columns=['artist_name', 'genre', 'topic'])

# Perform standard scaling on numerical features
scaled_numerical_features = standard_scaler.fit_transform(numerical_features)

# Replace original numerical features with scaled ones
transformed_data[numerical_features.columns] = scaled_numerical_features

transformed_data.head()
# PLOTTING HISTOGRAM

transformed_data.hist(bins=50, figsize=(20, 20))
plt.show()

tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
transformed_data = tsne.fit_transform(transformed_data)

x_train, x_test = train_test_split(transformed_data, test_size=.2)
x_train

# PLOTTING DATA POINTS

plt.figure(figsize=(30, 30))
plt.scatter(x_train[:, 0], x_train[:, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


model = DBSCAN(eps=4, min_samples=5)
clusters = model.fit_predict(x_train)
clusters

# VISUALIZING CLUSTERS

plt.figure(figsize=(30, 30))
plt.scatter(x_train[:, 0], x_train[:, 1],
            c=clusters, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()


calinski_harabasz = calinski_harabasz_score(x_train, clusters)
davies_bouldin = davies_bouldin_score(x_train, clusters)


print(f"Calinski-Harabasz Index: {calinski_harabasz}")
print(f"Davies-Bouldin Index: {davies_bouldin}")


# Assuming you have already trained a DBSCAN model named 'model2'

# Get cluster predictions for the test data
test_cluster_labels = model.fit_predict(x_test)

# Print cluster assignments for each test data point
for i, cluster_label in enumerate(test_cluster_labels):
    print(f"Test data point {i+1} belongs to cluster {cluster_label}")


def recommend_similar_songs(song_data, target_song_index, cluster_labels):
    """
    Recommends similar songs from the same cluster as the target song.

    Parameters:
    - song_data (numpy.ndarray or list): Array containing song data.
    - target_song_index (int): Index of the target song in the song data.
    - cluster_labels (numpy.ndarray or list): Array containing cluster labels for each song.

    Returns:
    - recommended_indices (list): List of indices of recommended songs.
    """
    # Find the cluster label of the target song
    target_cluster_label = cluster_labels[target_song_index]

    # Find indices of songs in the same cluster as the target song
    similar_song_indices = np.where(cluster_labels == target_cluster_label)[0]

    # Remove the target song index from the list of similar songs
    similar_song_indices = similar_song_indices[similar_song_indices !=
                                                target_song_index]

    return list(similar_song_indices)

# Example usage:
# Assuming you have your song data, target song index, and cluster labels
# recommended_indices = recommend_similar_songs(song_data, target_song_index, cluster_labels)
