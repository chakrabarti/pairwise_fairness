import numpy as np


def CreateDistMatrix(input_file):
    """
    Given a pmed file generates a distance matrix.
    """
    C = np.genfromtxt(input_file)

    # Number of points
    n = int(C[0][0])

    # Number of clusters
    num_clusters = int(C[0, 2])

    # Initialize distance matrix properly to use Floyd-Warshall
    dist_matrix = np.ones((n, n))*np.PINF
    for y in C[1:]:
        a, b, c = int(y[0]), int(y[1]), y[2]
        dist_matrix[a-1, b-1] = y[2]
        dist_matrix[b-1, a-1] = y[2]
    np.fill_diagonal(dist_matrix, 0)

    # uses Floyd-Warshall to compute all distances (since distances are not completely specified by the pmed files)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]

    return dist_matrix, num_clusters
