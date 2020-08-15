#!/usr/bin/env python
import numpy as np
from copy import deepcopy
from clustering.scr import kCenterScr
from clustering.gonzalez import Gonzalez, GonzalezPlus
from preprocess import CreateDistMatrix
from clustering.fairalg import *
import argparse
from time import time
from simhelp import *
import csv
from pathlib import Path
from typing import *
# Run simulations for adult dataset


def ConductSimulation(k: int, num_iters: int, output_CSV: str, output_folder: str, verbose: bool):
    with open('data/adult_dataset_subsampled.in', 'rb') as f:
        dist_matrix = np.load(f)

    # Modify dist_matrix so that only upper triangular part is considered (don't want to double count contributions)
    dist_matrix_modded = deepcopy(dist_matrix)
    dist_matrix_modded[np.tril_indices(np.shape(dist_matrix)[0])] = np.PINF

    # Run Scr Algorithm
    t0 = time()
    centers, scr_radius, radii, center_assignments = kCenterScr(dist_matrix, k)
    t1 = time()

    if verbose:
        print("It took {} to run kCenterScr on {}".format(t1-t0, "adult dataset"))

    # Calculate statistics for Scr
    ScrResults = UnfairCSVFormatterForAdult(
        'Scr', dist_matrix, dist_matrix_modded, scr_radius, scr_radius, center_assignments, verbose=verbose)

    if verbose:
        print(f"ScrRadius: {scr_radius}")

    if scale_file:
        scale_params = list(input_scales)
    else:
        scale_params = [scr_radius/16, scr_radius/4, scr_radius]

    num_params = len(scale_params)
    avg_sep_ratio = np.zeros(num_params)
    max_sep_ratio = np.zeros(num_params)
    avg_max_radius = np.zeros(num_params)
    overall_max_radius = np.zeros(num_params)
    avg_num_diff_clusters = np.zeros(num_params)
    max_num_diff_clusters = np.zeros(num_params)

    # Record statistics for Scr and fair algorithm
    with open(output_CSV, mode='w+') as f:
        f_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(["", "Parameter Value", "Average Separation Ratio", "Maximum Separation Ratio",
                           "Maximum Radius", "Average Number of Different Clusters", "Maximum Number of Different Clusters"])
        f_writer.writerow(ScrResults)
        f_writer.writerow(["", "Parameter Value", "Average Separation Ratio", "Maximum Separation Ratio", "Average Maximum Radius",
                           "Overall Maximum Radius", "Average Number of Different Clusters", "Maximum Number of Different Clusters"])
        for num in range(num_params):
            avg_sep_ratio[num], max_sep_ratio[num], avg_max_radius[num], overall_max_radius[num], avg_num_diff_clusters[num], max_num_diff_clusters[num] = PreservationSeparationFair(dist_matrix, dist_matrix_modded,
                                                                                                                                                                                      centers, radii, center_assignments, scr_radius, scale_params[
                                                                                                                                                                                          num],
                                                                                                                                                                                      num_iters, verbose=verbose)

            f_writer.writerow(['Scale Param ' + str(num+1), scale_params[num], avg_sep_ratio[num], max_sep_ratio[num],
                               avg_max_radius[num], overall_max_radius[num], avg_num_diff_clusters[num], max_num_diff_clusters[num]])


if __name__ == "__main__":

    # Command line option parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', dest="k", type=int,
                        default=2, help="number of clusters")
    parser.add_argument('--num_iters', dest="num_iters",
                        type=int, default=10000, help="number of iterations")
    parser.add_argument('--scale_file', dest="scale_file", type=str,
                        default=None, help="file to manually enter scale parameters")
    parser.add_argument('--out_folder', dest="out_folder", type=str,
                        default="AllAdultOutputs", help="where the output will store")
    parser.add_argument('--output_file', dest="output_file",
                        type=str, default=None, help="manually name output_file")
    parser.add_argument('-v', '--verbose', dest="verbose",
                        action='store_true', help="print stuff to stdout")
    parser.set_defaults(verbose=False)
    parameters = vars(parser.parse_args())

    scale_file = parameters["scale_file"]
    verbose = parameters['verbose']
    output_folder = parameters['out_folder']
    output_file = parameters['output_file']
    k = parameters['k']
    num_iters = parameters['num_iters']

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if (scale_file):
        with open(scale_file, "r") as f:
            scale = csv.reader(f, delimiter=",")
            for row in scale:
                input_scales = row
                input_scales = (float(x) for x in input_scales)

    if (not parameters['output_file']):
        output_file = f"adult_dataset_{k}"
    output_CSV = f"{output_folder}/{output_file}.csv"

ConductSimulation(k, num_iters, output_CSV, output_folder, verbose)
