1. Normalize features of the dataset and ensure every feature is
   comparable in scale to every other feature.

2. Compute the measure of spread of the entire dataset using the
   average distance between nodes.

3. Define the density radius r0 based on the spread of the dataset:
   r0 = (average distance) × α
   where α = 0.2 by default (can be adjusted).
   Use a larger r0 for widely spread data and a smaller r0 for
   tightly clustered data.

4. For each instance xi in dataset X:
     - Examine the neighborhood around xi with radius r0
     - Count how many points fall inside this neighborhood
     - Assign this count as the density value for xi

5. Select the point with the highest density value as the first centroid.

6. Until k centroids are selected:
     - Compute the distance of each node to its closest centroid
     - For each node, compute: density × distance²
     - Select the node with the highest value
       (K-means++-style selection)

7. Use the selected centroids as initialization and run the
   standard K-means algorithm.
