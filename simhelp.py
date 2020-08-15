#!/usr/bin/env python

import numpy as np
from copy import deepcopy
from clustering.scr import kCenterScr
from clustering.gonzalez import Gonzalez, GonzalezPlus
from clustering.fairalg import *
import argparse
from time import time
import csv
from typing import *


def PreservationSeparationPreproc(dist_matrix: np.array, dist_matrix_modded: np.array, scr_radius: float) -> Tuple[np.array, float, np.array, np.array]:
    """
    Function which does preprocessing for ratio calculations
    Inputs:
        1. dist_matrix: matrix with distances between every point
        2. dist_matrix_modded: dist_matrix but with lower triangle set to positive infinity
        3. scr_radius: max radius using Scr algorithm
    Outputs:
        1. interesting_locs: locations where the vertices are close enought that we care about calculating separation
        2. alpha: the probability bound that we will be comparing to
        3. total_count: a zeroed matrix where we will be checking/storing if vertices are clustered together
        4. num_diff_cluster_array: a zeroed array where we will be storing the number of different clusters that vertices within a certain bound exist in
                                in this case the bound is scr_radius/4
    """

    # We only care about locations where the distance between the vertices is within max_radius
    interesting_locs = np.argwhere(dist_matrix_modded <= scr_radius)

    # Calculate what the alpha bound is for each pair of vertices (note that the alpha bound is just the distance multipled by the rate parameter)
    alpha = dist_matrix/scr_radius
    np.fill_diagonal(alpha, 1)

    total_count = np.zeros(np.shape(dist_matrix))
    num_diff_clusters_array = np.zeros(np.shape(dist_matrix)[0])

    return interesting_locs, alpha, total_count, num_diff_clusters_array


def PreservationSeparationCore(interesting_locs: np.array, alpha: float, total_count: np.array, num_diff_clusters_array: np.array, dist_matrix: np.array, dist_matrix_modded: np.array, scr_radius: float, center_assignments: np.array) -> None:
    """
    Function which does ratio calculations
    Inputs:
        1. interesting_locs: locations where the vertices are close enought that we care about calculating separation
        2. alpha: the probability bound that we will be comparing to
        3. total_count: a zeroed matrix where we will be checking/storing if vertices are clustered together
        4. num_diff_clusters_array: a zeroed array where we will be storing the number of different clusters that vertices within a certain bound exist in
                                in this case the bound is scr_radius/4
        5. dist_matrix: matrix with distances between every point
        6. dist_matrix_modded: dist_matrix but with lower triangle set to positive infinity
        7. scr_radius: max radius using Scr algorithm
        8. center_assignments: array with the assignments of vertices to centers

    """
    for x in interesting_locs:
        if (center_assignments[x[0]] != center_assignments[x[1]]):
            total_count[x[0], x[1]] += 1

    for j in range(dist_matrix.shape[0]):
        locs = np.argwhere(dist_matrix[j] <= scr_radius/4)
        num_diff_clusters = len(np.unique(center_assignments[locs]))
        num_diff_clusters_array[j] = num_diff_clusters


def PreservationSeparationPostprocFair(interesting_locs: np.array, alpha: float, total_count: np.array, num_diff_clusters_matrix: np.array, num_iters: int) -> Tuple[float, float, float, float]:
    """
    Function which does postprocessing for ratio calculations if running the fair algorithm (different metrics for interest for fair vs. unfair algorithm)
    Inputs:
        1. interesting_locs: locations where the vertices are close enought that we care about calculating separation
        2. alpha: the probability bound that we will be comparing to
        3. total_count: a matrix which contains the counts for vertices that were not clustered together
        4. num_diff_clusters_matrix: a matrix which contains the number of different clusters that vertices within a certain bound exist in
                                in this case the bound is scr_radius/4 (the first dimension is the the iteration for which the data was recorded)
        5. num_iters: number of iteratations for which to run fair algorithm
    Outputs:
        1. avg_sep_ratio: average separation ratio for all vertices
        2. max_sep_ratio: maximum separation ratio among all vertices
        3. avg_num_diff_clusters: the average number of different clusters for each vertex (over all iterations)
        4. max_num_diff_clusters: maximum number of different clusters across averages for each vertex (over all iterations)
    """

    avg_num_diff_clusters = np.average(num_diff_clusters_matrix)
    max_num_diff_clusters = np.amax(
        np.average(num_diff_clusters_matrix, axis=0))
    # Compute the realized separation "probabilities"
    actual_separation = (total_count/num_iters)
    # Compute the ratio between realized and theoretical bound
    actual_sep_ratios = actual_separation/alpha
    avg_sep_ratio = np.sum(actual_sep_ratios)/len(interesting_locs)
    max_sep_ratio = np.max(actual_sep_ratios)

    return avg_sep_ratio, max_sep_ratio, avg_num_diff_clusters, max_num_diff_clusters


def PreservationSeparationPostprocUnfair(alpha: float, total_count: np.array, num_diff_clusters_array: np.array) -> Tuple[float, float, float, float]:
    """
    Function which does postprocessing for ratio calculations if running the unfair algorithm (different metrics for interest for fair vs. unfair algorithm)
    Inputs:
        1. alpha: the probability bound that we will be comparing to
        2. total_count: a matrix which contains the counts for vertices that were not clustered together
        3. num_diff_clusters_array: a zeroed matrix where we will be storing the number of different clusters that vertices within a certain bound exist in
                                in this case the bound is scr_radius/4
    Outputs:
        1. avg_sep_ratio: average separation ratio for all vertices
        2. max_sep_ratio: maximum separation ratio among all vertices
        3. avg_num_diff_clusters: the average number of different clusters for each vertex
        4. max_num_diff_clusters: maximum number of different clusters across averages for each vertex
    """

    avg_num_diff_clusters = np.average(num_diff_clusters_array)
    max_num_diff_clusters = np.amax(num_diff_clusters_array)
    sepRatio = total_count/alpha
    avg_sep_ratio = np.average(sepRatio)
    max_sep_ratio = np.amax(sepRatio)

    return avg_sep_ratio, max_sep_ratio, avg_num_diff_clusters, max_num_diff_clusters


def PreservationSeparationFair(dist_matrix: np.array, dist_matrix_modded: np.array, centers: np.array, radii: np.array, center_assignments: np.array, scr_radius: float, scale_param: float, num_iters: int, verbose=False) -> Tuple[float, float, float, float, float, float]:
    """
    Function which does preprocessing, core for number of iterations, and post processing for fair algorithm
    Inputs:
        1. dist_matrix: matrix with distances between every point
        2. dist_matrix_modded: dist_matrix but with lower triangle set to positive infinity
        3. centers: original centers (before running fair algorithm)
        4. radii: radii of clusters (before running fair algorithm)
        5. center_assignments: original center assignments (before running fair algorithm)
        6. scr_radius: max radius using Scr algorithm
        7. scale_param: parameter to be used for exponential for fair algorithm
        8. num_iters: number of iterations to run fair algorithm
        9. verbose: Boolean variable indicating whether to print out times
    Outputs:
        1. avg_sep_ratio: average separation ratio for all vertices
        2. max_sep_ratio: maximum separation ratio among all vertices
        3. avg_max_radius: average maximum radius among all iterations
        4. overall_max_radius: overall maximum radius (among max radii for all iterations)
        5. avg_num_diff_clusters: the average number of different clusters for each vertex (over all iterations)
        6. max_num_diff_clusters: maximum number of different clusters across averages for each vertex (over all iterations)
    """
    t0 = time()

    interesting_locs, alpha, total_count, num_diff_clusters_array = PreservationSeparationPreproc(
        dist_matrix, dist_matrix_modded, scr_radius)

    maxRadii = np.zeros(num_iters)

    num_diff_clusters_matrix = np.zeros((num_iters, np.shape(dist_matrix)[0]))

    for i in range(num_iters):
        maxRadii[i], current_iter_center_assignments = ComputeFairClusters(
            dist_matrix, centers, radii, center_assignments, scale_param)
        PreservationSeparationCore(interesting_locs, alpha, total_count, num_diff_clusters_array,
                                   dist_matrix, dist_matrix_modded, scr_radius, current_iter_center_assignments)
        num_diff_clusters_matrix[i] = num_diff_clusters_array

    avg_sep_ratio, max_sep_ratio, avg_num_diff_clusters, max_num_diff_clusters = PreservationSeparationPostprocFair(
        interesting_locs, alpha, total_count, num_diff_clusters_matrix, num_iters)

    avg_max_radius = np.average(maxRadii)
    overall_max_radius = np.amax(maxRadii)

    t1 = time()
    if verbose:
        print(f"Time for ratio calculations for fair algorithm: {t1-t0}")

    return avg_sep_ratio, max_sep_ratio, avg_max_radius, overall_max_radius, avg_num_diff_clusters, max_num_diff_clusters


def PreservationSeparationUnfair(dist_matrix: np.array, dist_matrix_modded: np.array, scr_radius: float, center_assignments: np.array, verbose: bool = False) -> Tuple[float, float, float, float]:
    """
    Function which does preprocessing, core once, and post processing for unfair algorithm
    Inputs:
        1. dist_matrix: matrix with distances between every point
        2. dist_matrix_modded: dist_matrix but with lower triangle set to positive infinity
        3. scr_radius: max radius using Scr algorithm
        4. center_assignments: original center assignments (before running fair algorithm)
        5. verbose: Boolean variable indicating whether to print out times
    Outputs:
        1. avg_sep_ratio: average separation ratio for all vertices
        2. max_sep_ratio: maximum separation ratio among all vertices
        3. avg_num_diff_clusters: the average number of different clusters for each vertex (over all iterations)
        4. max_num_diff_clusters: maximum number of different clusters across averages for each vertex (over all iterations)
    """
    t0 = time()

    interesting_locs, alpha, total_count, num_diff_clusters_array = PreservationSeparationPreproc(
        dist_matrix, dist_matrix_modded, scr_radius)

    PreservationSeparationCore(interesting_locs, alpha, total_count, num_diff_clusters_array,
                               dist_matrix, dist_matrix_modded, scr_radius, center_assignments)

    avg_sep_ratio, max_sep_ratio, avg_num_diff_clusters, max_num_diff_clusters = PreservationSeparationPostprocUnfair(
        alpha, total_count, num_diff_clusters_array)

    t1 = time()
    if verbose:
        print(f"Time for ratio calculations for unfair algorithm: {t1-t0}")

    return avg_sep_ratio, max_sep_ratio, avg_num_diff_clusters, max_num_diff_clusters


def UnfairCSVFormatterForPmed(name: str, dist_matrix: np.array, dist_matrix_modded: np.array, opt_radius: float, scr_radius: float, max_radius: float, center_assignments: np.array, verbose: bool = False) -> List[Union[str, float]]:
    """
    Function to format unfair algorithm metrics in CSV format for pmed experiments
    """
    avg_sep_ratio, max_sep_ratio, avg_num_diff_clusters, max_num_diff_clusters = PreservationSeparationUnfair(
        dist_matrix, dist_matrix_modded, scr_radius, center_assignments, verbose=verbose)
    return ['Unfair Clustering Method', name, avg_sep_ratio, max_sep_ratio, max_radius/opt_radius, max_radius, avg_num_diff_clusters, max_num_diff_clusters]


def UnfairCSVFormatterForAdult(name: str, dist_matrix: np.array, dist_matrix_modded: np.array, scr_radius: float, max_radius: float, center_assignments: np.array, verbose: bool = False) -> List[Union[str, float]]:
    """
    Function to format unfair algorithm metrics in CSV format for adult experiments
    """
    avg_sep_ratio, max_sep_ratio, avg_num_diff_clusters, max_num_diff_clusters = PreservationSeparationUnfair(
        dist_matrix, dist_matrix_modded, scr_radius, center_assignments, verbose=verbose)
    return ['Unfair Clustering Method', name, avg_sep_ratio, max_sep_ratio, max_radius, avg_num_diff_clusters, max_num_diff_clusters]
