import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import make_blobs

data = loadData()

keys = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)', 'species', 'species_id']

X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)']]
y = data["species_id"]

species = data["species"].unique()
print("Species:", species)

# Create a PCA object with the desired number of components
n_components = 2  # Adjust this number to the desired dimensionality
pca = PCA(n_components=n_components)

# Fit the PCA model to your data and transform it
X_reduced = pca.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                             test_size=0.3,random_state=312)

# PLOT data
df = pd.DataFrame(X_reduced)
df["species"] = data["species"]
df["species_id"] = data["species_id"]

# Define colors for each class
colors = {0: 'red', 1: 'blue', 2: 'green'}

# Create a scatter plot with different colors for each class
plt.figure(figsize=(8, 6))
for label, color in colors.items():
    subset = df[df['species_id'] == label]
    plt.scatter(subset[0], subset[1], c=color, label=f'{species[label]}', marker='o', s=50)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Sepal Data after PCA')
plt.grid(True)

# Clustering with K-Means
kmeans = KMeans(n_clusters=3, n_init="auto").fit(X_reduced)
labels = kmeans.labels_
df["kmeans predictions"] = labels

# Get cluster means (centroids)
cluster_centers = kmeans.cluster_centers_

# Calculate distances between data points and cluster means
distances = []
for i in range(len(X_reduced)):
    cluster_label = labels[i]
    cluster_mean = cluster_centers[cluster_label]
    distance = np.linalg.norm(X_reduced[i] - cluster_mean)  # Euclidean distance
    distances.append(distance)


# Create a scatter plot showing centroids and obtained cluster points
plt.figure(figsize=(8, 6))
for label, color in colors.items():
    subset = df[df["kmeans predictions"] == label]
    plt.scatter(subset[0], subset[1], c=color, 
                label=f'Species No. {label+1}', marker='o', s=50)

# Plot cluster means (centroids)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
            c='black', s=200, marker='X', label='Cluster Means')

# Optional: Plot lines connecting data points to cluster means
for i in range(len(X)):
    plt.plot([X_reduced[i, 0], 
            cluster_centers[labels[i], 0]], [X_reduced[i, 1], 
            cluster_centers[labels[i], 1]], 'k-', alpha=0.3)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering with Cluster Means and Distances')
plt.legend()
plt.grid(True)

plt.show()