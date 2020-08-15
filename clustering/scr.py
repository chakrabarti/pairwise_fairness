import numpy as np
from operator import itemgetter
from itertools import groupby
from copy import deepcopy
from typing import *


def ExtractCostsAdjRepr(dist_matrix: np.array) -> Tuple[List[float], List[Tuple[int, int, float]]]:
    """
    Extract the desired representation in order to run the Scr algorithm.
    """
    n = np.shape(dist_matrix)[0]

    indices = [(x, y) for x in range(n) for y in range(n)]
    flattened = list(dist_matrix.reshape(dist_matrix.size))
    combined = list(zip(indices, flattened))
    adj_rep = [(x, y, z) for ((x, y), z) in combined]

    costs = sorted(list(set([z for (x, y, z) in adj_rep])))
    # don't want to include "self edges" so we remove the zero cost edge
    costs = costs[1:]

    return costs, adj_rep


def PruneGraph(graph: np.array, prune_cost: float) -> np.array:
    """
    Prune the graph to only include edges with costs lower than prune_cost.
    """
    filtered_graph = [(x, y, z) for (x, y, z) in graph if z <= prune_cost]
    return filtered_graph


def SortByVertex(pruned: np.array) -> Dict[int, List[Tuple[int, float]]]:
    """
    Sorts graph by vertex.
    """
    sortkeyfn = itemgetter(0)
    result = {}
    for key, valuesiter in groupby(pruned, key=sortkeyfn):
        result[key] = list((v[1], v[2]) for v in valuesiter)
    return result


def Scr(graph_dict: Dict[int, List[Tuple[int, float]]]):
    """ 
    Runs Scr algorithm.
    """
    CovCnt = {}
    for key in graph_dict.keys():
        CovCnt[key] = len(graph_dict[key])
    score = deepcopy(CovCnt)
    V = len(CovCnt)
    D = []
    for i in range(V):
        next_vertex = min(score, key=score.get)
        vertexFound = False
        for (y, dist) in graph_dict[next_vertex]:
            if CovCnt[y] == 1:
                vertexFound = True
                break
        if vertexFound:
            D.append(next_vertex)
            for (y, dist) in graph_dict[next_vertex]:
                CovCnt[y] = 0
        else:
            for (y, dist) in graph_dict[next_vertex]:
                if CovCnt[y] > 0:
                    CovCnt[y] -= 1
                    score[y] += 1
        score[next_vertex] = np.PINF

    return D


def kCenterScr(dist_matrix: np.array, k: int):
    """
    The k-center version of the Scr algorithm.
    """
    costs, adj_rep = ExtractCostsAdjRepr(dist_matrix)
    for cost in costs:
        pruned = PruneGraph(adj_rep, cost)
        sorted_V = SortByVertex(pruned)
        D = Scr(sorted_V)
        if len(D) <= k:
            centers = np.array(D)
            relevant_distances = dist_matrix[:, D]
            center_assignments = centers[np.argmin(relevant_distances, axis=1)]
            radii = np.amin(relevant_distances, axis=1)
            max_radius = np.amax(radii)
            clusters = list(set(center_assignments))
            return centers, max_radius, radii, center_assignments
