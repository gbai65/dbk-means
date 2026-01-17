from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

trainingData = pd.read_csv('skewedData.csv')
inputTrain = trainingData.iloc[:, :-1].copy().values
outputTrain = trainingData.iloc[:, -1].copy().values

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0, n_init='auto')
kmeans.fit(inputTrain)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.figure(figsize=(10, 6))

for i in range(2):
    cluster_points = inputTrain[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', alpha=0.7, s=50)

plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids', linewidths=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering Results (2 Clusters)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()