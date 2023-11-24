import numpy as np
from cuml.neighbors import NearestNeighbors

# Assuming X is your data with shape [6400, 7]
X = np.random.rand(6400, 7)  # Replace this with your actual data

# Set the number of neighbors for the adjacency matrix
n_neighbors = 5  # You can adjust this based on your requirements

# Build the k-d tree on the GPU
knn_gpu = NearestNeighbors(n_neighbors=n_neighbors)
knn_gpu.fit(X)

# Query the nearest neighbors for each point on the GPU
distances, indices = knn_gpu.kneighbors(X)

# Display the adjacency matrix
print(indices)