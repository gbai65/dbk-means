import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score, silhouette_samples
import time

starttime = time.time()
plotNum=0


def dbkMeans(input, k):
    r0 = calculateR0(input)
    distMatrix = cdist(input, input, metric = "euclidean")
    localDensities = np.maximum(calculateLocalDensity(input, distMatrix, r0), 1e-10)
    selectedCentroidsIxs = selectCentroids(localDensities, distMatrix, k)
    selectedCentroidsPos = [list(input[ix]) for ix in selectedCentroidsIxs]
    selectedCentroidsPos = np.array(selectedCentroidsPos, dtype=np.float64)
    plot(input, selectedCentroidsPos, np.array([0 for i in range(len(input))], dtype=int),1)

    kmeans = KMeans(n_clusters = k, init=selectedCentroidsPos, n_init=1)
    kmeans.fit(input)

    centroids= kmeans.cluster_centers_
    labels = kmeans.labels_
    plot(input, centroids, labels, k)
    
def selectCentroids(localDensities, distMatrix, k): #density-based k-means++
    centroidIxs = []
    firstIx = max(range(len(localDensities)), key = localDensities.__getitem__)
    centroidIxs.append(firstIx)
    for iter in range(k-1):
        minDistCentroid = np.min(distMatrix[:, centroidIxs], axis=1)
        minDistCentroid[centroidIxs] = 0
        centroidIxs.append(np.argmax(localDensities*minDistCentroid**2))
    return centroidIxs

def calculateLocalDensity(input, distanceMatrix, r0):
    densities = np.sum(distanceMatrix<= r0, axis=1) #use axis to sum for each indiv. point
    return densities

def calculateR0(input): #based on data spread
    numSamples = len(input)
    sampleIdx = np.random.choice(numSamples, min(1000, numSamples), replace=False) #take a sample if dataset too big
    sampleDist = pdist(input[sampleIdx], metric= "euclidean") #FOR LATER: USE CDIST HERE
    avgDist = np.mean(sampleDist) #calculate r0 based off average distances between nodes
    alpha = 0.2 #we should test what alpha val to use with trial-and-error
    return alpha * avgDist

def normalizeFeatures(input): #scikit-learn z-score normalization; (x - mean)/stdDev
    sl = StandardScaler()
    return sl.fit_transform(input), sl

def normalKMeans(input, k):
    initialCntrds = np.array([input[i] for i in range(k)]) 
    plot(input, initialCntrds, np.array([0 for i in range(len(input))], dtype=int), k)
    kmeans = KMeans(n_clusters = k, init= initialCntrds, n_init=1)
    kmeans.fit(input)
    plot(input, kmeans.cluster_centers_, kmeans.labels_, k)

def kMeansPP(input, k):
    initialKMeansPP = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=0)
    initialKMeansPP.fit(input)
    initialKMPP = initialKMeansPP.cluster_centers_
    plot(input, initialKMPP, np.array([0 for i in range(len(input))], dtype=int), k)
    fullKMeansPP = KMeans(n_clusters=k, init='k-means++', random_state=0, n_init='auto')
    fullKMeansPP.fit(input)
    plot(input, fullKMeansPP.cluster_centers_, fullKMeansPP.labels_, k)

def plot(input, centroids, labels, k): #make a pretty looking plot
    global plotNum
    plt.figure(figsize=(10, 6))
    for i in range(k):
        cluster_points = input[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", alpha=0.7, s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="X", s=200, label="Centroid", linewidths=2)
    plt.xlabel("Attribute 1")
    plt.ylabel("Attribute 2")
    plt.title(f"Clustering Results: {["DBK-Means Initial Centroids", "DBK-Means Final Clusters", "K-Means Initial Centroids", "K-Means Final Centroids", "K-Means++ Initial Centroids", "K-Means Final Centroids"][plotNum]}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"dbkPlot{plotNum}.png")
    plotNum+=1

def main(k):
    trainingData = pd.read_csv("skewedData.csv")
    trainingData = trainingData.drop(columns=["label"])
    trainingData = trainingData.values
    trainingData = normalizeFeatures(trainingData)[0]
    print(f"DBKMeans commencing; {round(time.time()-starttime, 3)}s elapsed")
    dbkMeans(trainingData, k)
    print(f"DBKMeans finished; {round(time.time()-starttime, 3)}s elapsed")
    print(f"KMeans commencing; {round(time.time()-starttime, 3)}s elapsed")
    normalKMeans(trainingData, k)
    print(f"KMeans finished; {round(time.time()-starttime, 3)}s elapsed")
    print(f"KMeans++ commencing; {round(time.time()-starttime, 3)}s elapsed")
    kMeansPP(trainingData, k)
    print(f"KMeans++ finished; {round(time.time()-starttime, 3)}s elapsed")




main(10)


