import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score, silhouette_samples

trainingData = pd.read_csv("skewedData.csv").values

def dbkMeans(input, k):
    input, scaler = normalizeFeatures(input)
    numSamples, numFeatures = input.shape

    r0 = calculateR0(input)
    #might be a problem?
    distMatrix = cdist(input, input, metric = "euclidean")
    localDensities = [ea1 for ea in calculateLocalDensity(input, distMatrix, r0) if ((ea1:=ea) or (ea1:=1e-10))] #sorry aneesh
    selectedCentroidsIxs = selectCentroids(localDensities, distMatrix, k)
    return 
    
def selectCentroids(localDensities, k): #density-based k-means++
    centroidIxs = []
    firstIx = max(range(len(localDensities)), key = localDensities.__getitem__)
    centroidIxs.append(firstIx)
    for iter in range(k-1):
        #calculate dist from data points to existing centers
        #choose next center 
    return

def calculateLocalDensity(input, distanceMatrix, r0):
    densities = np.sum(distanceMatrix<= r0, axis=1) #use axis to sum for each indiv. point
    return densities

def calculateR0(input): #based on data spread
    numSamples = len(input)
    sampleIdx = np.random.choice(numSamples, numSamples, replace=False)
    sampleDist = pdist(input[sampleIdx], metric= "euclidean") #FOR LATER: USE CDIST HERE
    avgDist = np.mean(sampleDist) #calculate r0 based off average distances between nodes
    alpha = 0.02 #we should test what alpha val to use with trial-and-error
    return alpha * avgDist

def normalizeFeatures(input): #scikit-learn z-score normalization; (x - mean)/stdDev
    sl = StandardScaler()
    return sl.fit_transform(input), sl

def plot(input, centroids, labels): #make a pretty looking plot
    plt.figure(figsize=(10, 6))
    for i in range(2):
        cluster_points = input[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", alpha=0.7, s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="X", s=200, label="Centroid", linewidths=2)
    plt.xlabel("Attribute 1"); plt.ylabel("Attribute 2"); plt.title(f"Clustering Results")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig("silPlot.png")

# kMeans(trainingData)


