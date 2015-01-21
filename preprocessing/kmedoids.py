
# This file came from https://github.com/salspaugh/machine_learning/blob/master/clustering/kmedoids.py


import numpy as np
import random


# *_medoids[k] is the index of the instance that is the medoid for the kth cluster
# clusters[n] is the cluster assignment in [0,k-1] for the nth instance
def cluster(distances, k=3, maxIters=100000, indent=2):

	m = distances.shape[0] # number of points

	# Pick k random medoids.
	curr_medoids = np.array([-1]*k)
	while not len(np.unique(curr_medoids)) == k:
		curr_medoids = np.array(random.sample(xrange(m), k))
	old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
	new_medoids = np.array([-1]*k)
   
	iter_num = 1
	# Until the medoids stop updating, do the following:
	while not ((old_medoids == curr_medoids).all()):
		# Assign each point to cluster with closest medoid.
		print "%sIteration %d" % (indent * "\t", iter_num)
		iter_num += 1
		clusters = assign_points_to_clusters(curr_medoids, distances)

		# Update cluster medoids to be lowest cost point. 
		for curr_medoid in curr_medoids:
			cluster = np.where(clusters == curr_medoid)[0]
			new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

		old_medoids[:] = curr_medoids[:]
		curr_medoids[:] = new_medoids[:]
		if iter_num > maxIters:
			break

	return clusters, curr_medoids

def assign_points_to_clusters(medoids, distances):
	distances_to_medoids = distances[:,medoids]
	return medoids[np.argmin(distances_to_medoids, axis=1)]

def compute_new_medoid(cluster, distances):
	cluster_idxs = np.ix_(cluster, cluster)
	#cluster_distances = np.ma.masked_array(data=distances, mask=_mask, fill_value=10e9)
	cluster_distances = distances[cluster_idxs]
	costs = cluster_distances.sum(axis=1)
	return cluster[costs.argmin(axis=0)]
