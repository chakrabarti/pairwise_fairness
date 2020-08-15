import numpy as np
from copy import deepcopy
import argparse
from typing import *


def ChooseInitialMin(A: np.array, k: int) -> float:
    """
    Takes minimum of maximum of each row of matrix.
    """
    max_each_row = np.amax(A, axis=1)
    minmax = np.argmin(max_each_row)
    return minmax


def Rest(A: np.array, k: int, centers: np.array) -> np.array:
    """
    Determines rest of centers to be used in Gonzalez.
    """
    while k > 0:
        A[centers, :] = np.zeros((np.size(centers), np.shape(A)[1]))
        relevant = A[:, centers]
        min_each_row = np.amin(relevant, axis=1)
        maxmin = np.argmax(min_each_row)
        centers.append(maxmin)
        k -= 1
    return np.array(centers)


def GonzalezVariant(choose_function: Callable[[np.array, int], float], dist_matrix: np.array, k: int) -> Tuple[np.array, float, np.array, np.array, List[int]]:
    """
    Helper for the Gonzalez algorithm.    
    """
    A = deepcopy(dist_matrix)
    centers = []
    centers.append(choose_function(A, k))
    centers = Rest(A, k-1, centers)
    relevant_distances = dist_matrix[:, centers]
    center_assignments = centers[np.argmin(relevant_distances, axis=1)]
    radii = np.amin(relevant_distances, axis=1)
    max_radius = np.amax(radii)
    clusters = list(set(center_assignments))

    return centers, max_radius, radii, center_assignments, clusters


def Gonzalez(dist_matrix: np.array, k: int) -> Tuple[np.array, float, np.array, np.array, List[int]]:
    """
    Run Gonzalez algorithm.
    """
    return GonzalezVariant(ChooseInitialMin, dist_matrix, k)


def GonzalezPlus(dist_matrix: np.array, k: int) -> Tuple[np.array, float, np.array, np.array, List[int]]:
    """
    Run Gonzalez Plus algorithm.
    """
    N = np.shape(dist_matrix)[0]
    max_radii = np.zeros(N)
    for i in range(N):
        max_radii[i] = GonzalezVariant((lambda A, k: i), dist_matrix, k)[1]
    best = np.argmin(max_radii)
    return GonzalezVariant((lambda A, k: best), dist_matrix, k)
