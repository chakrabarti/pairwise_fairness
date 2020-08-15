import numpy as np
from copy import deepcopy
from time import time
from typing import *


def ComputeFairClusters(dist_matrix: np.array, centers: np.array, radii: np.array, center_assignments: np.array, scale_param: float = 1.0):
    """ Algorithm to produce fair clusters with alpha probability preservation guarantees.
    Inputs:
        1. dist_matrix: matrix representation of the distances between each pair of vertices
        2. centers: current designated centers of clusters computed by some clustering heuristic
        3. radii: vector representation of the distance from each vertex to the center of its assigned cluster
        4. center_assignments: vector representation of the vertex ID of the center for the assigned cluster for each vertex
        5. scale_param: beta parameter to be used to generate realization of exponential random variable;
        note that 1/beta = lambda (where lambda is the rate parameter)
    Outputs:
        1. new_max_radius: the maximum radius that results from the new clustering
        2. new_center_assignments: the center assignments which result from the new clustering
    """
    centers = np.array(centers)
    centers_matrix = dist_matrix[centers, :][:, centers]
    np.fill_diagonal(centers_matrix, np.inf)
    min_center_distance = np.min(centers_matrix)

    # Keeps track of the vertices that have been already clustered under the new clustering
    clustered_boolean = np.zeros(dist_matrix.shape[0], dtype=bool)
    new_centers = []  # Keeps track of the new centers
    new_clusters = {}  # Keeps track of assignment of vertices to new centers
    # Keeps track of new center assignments
    new_center_assignments = -1 * np.ones(dist_matrix.shape[0])
    # Keeps track of new radii
    new_radii = -1 * np.ones(dist_matrix.shape[0])

    # Shuffle centers randomly
    np.random.shuffle(centers)

    rand_vars = None

    # Iterate over all the old cluster centers
    for i, old_center in enumerate(centers):
        # Filter to only consider vertices that are in the cluster associated with the current old center
        locs = np.where(old_center == center_assignments)

        # Find radius of current old cluster
        cluster_radius = np.max(radii[locs])

        rand_var = np.random.exponential(scale_param)
        new_radius = cluster_radius + rand_var  # The new "effective radius"

        # Keeps track of vertices to be assigned to new cluster being created
        current_new_cluster = []
        # Check if the center of old clustered being considered has already been added
        need_new_center = clustered_boolean[old_center]
        # to a new cluster

        # Iterate through all vertices to check which can be potentially added to the new cluster
        current_new_cluster = (~clustered_boolean) & (
            dist_matrix[old_center] <= new_radius)
        clustered_boolean[current_new_cluster] = True

        if np.sum(current_new_cluster) > 0:  # Check that new cluster is non-empty
            clusterdist_matrix = dist_matrix[current_new_cluster,
                                             :][:, current_new_cluster]
            min_index = clusterdist_matrix.max(axis=1).argmin(axis=0)

            # Based on whether a new center is needed or not assign a new_center
            new_center = current_new_cluster.nonzero()[0][min_index]

            # Set the new center assignments and radius for the vertices added to the new cluster
            new_center_assignments[current_new_cluster] = new_center
            new_radii[current_new_cluster] = dist_matrix[new_center,
                                                         :][current_new_cluster]
            new_centers.append(new_center)

    # Compute new_max_radius
    new_max_radius = np.amax(new_radii)
    assert(np.all(new_center_assignments >= 0))
    assert(np.all(new_radius >= 0))
    return new_max_radius, new_center_assignments
