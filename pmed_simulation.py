#!/usr/bin/env python
import numpy as np
from copy import deepcopy
from clustering.scr import kCenterScr
from clustering.gonzalez import Gonzalez, GonzalezPlus
from process_pmed import CreateDistMatrix
from clustering.fairalg import *
import argparse
from time import time
from simhelp import *
from pathlib import Path
import csv

# Run simulations on pmed dataset


def ConductSimulation(num_iters: int, input_file_number: int, output_CSV: str, output_folder: str, verbose: bool = False):
    optimal = np.genfromtxt("data/pmed_optimal.txt")
    opt_radius = optimal[input_file_number-1]  # Due to zero indexing in Python
    if verbose:
        print("The optimal radius is {}".format(opt_radius))

    t0 = time()
    dist_matrix, k = CreateDistMatrix(input_file)
    t1 = time()
    if verbose:
        print("It took {} to read in {}".format(t1-t0, input_file))

    # Modify dist_matrix so that only upper triangular part is considered (don't want to double count contributions)
    dist_matrix_modded = deepcopy(dist_matrix)
    dist_matrix_modded[np.tril_indices(np.shape(dist_matrix)[0])] = np.PINF

    # Run Scr Algorithm
    t0 = time()
    centers, scr_radius, radii, center_assignments = kCenterScr(dist_matrix, k)
    t1 = time()

    if verbose:
        print("It took {} to run kCenterScr on {}".format(t1-t0, input_file))

    # Calculate statistics for Scr
    Scr_Results = UnfairCSVFormatterForPmed(
        'Scr', dist_matrix, dist_matrix_modded, opt_radius, scr_radius, scr_radius, center_assignments, verbose=verbose)

    # Run Gonzalez Algorithm
    t0 = time()
    centers, max_radius, radii, center_assignments, clusters = Gonzalez(
        dist_matrix, k)
    t1 = time()

    if verbose:
        print("It took {} to run Gonzalez on {}".format(t1-t0, input_file))

    # Calculate statistics for Gonzalez
    GonzalezResults = UnfairCSVFormatterForPmed(
        'Gonzalez', dist_matrix, dist_matrix_modded, opt_radius, scr_radius, max_radius, center_assignments, verbose=verbose)

    # Run GonzalezPlus Algorithm
    t0 = time()
    centers, max_radius, radii, center_assignments, clusters = GonzalezPlus(
        dist_matrix, k)
    t1 = time()

    if verbose:
        print("It took {} to run GonzalezPlus on {}".format(t1-t0, input_file))

    # Calculate statistics for GonzalezPlus
    GonzalezPlusResults = UnfairCSVFormatterForPmed(
        'GonzalezPlus', dist_matrix, dist_matrix_modded, opt_radius, scr_radius, max_radius, center_assignments, verbose=verbose)

    # Use default scale parameters if they aren't provided
    if scale_file:
        scale_params = list(input_scales)
    else:
        # Scale parameters on a base 2 log scale down from perfect separation ratio
        scale_params = [scr_radius/16, scr_radius/8,
                        scr_radius/4, scr_radius/2, scr_radius]
        # Old parameters, do not delete
        # scale_params = [1/((6*np.log(10) + np.log(k))/(scr_radius * 1)), 1/((6*np.log(10) + np.log(k))/(scr_radius * 3)), scr_radius]

    num_params = len(scale_params)
    avg_sep_ratio = np.zeros(num_params)
    max_sep_ratio = np.zeros(num_params)
    avg_max_radius = np.zeros(num_params)
    overall_max_radius = np.zeros(num_params)
    avg_num_diff_clusters = np.zeros(num_params)
    max_num_diff_clusters = np.zeros(num_params)

    with open(output_CSV, mode='w+') as f:
        f_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(["", "Parameter Value", "Average Separation Ratio", "Maximum Separation Ratio", "Max to Optimal Ratio",
                           "Maximum Radius", "Average Number of Different Clusters", "Maximum Number of Different Clusters"])
        f_writer.writerow(Scr_Results)
        f_writer.writerow(GonzalezResults)
        f_writer.writerow(GonzalezPlusResults)
        f_writer.writerow(["", "Parameter Value", "Average Separation Ratio", "Maximum Separation Ratio", "Average Max to Opt Radius Ratio", "Overall Max to Opt Radius Ratio",
                           "Average Maximum Radius", "Overall Maximum Radius", "Average Number of Different Clusters", "Maximum Number of Different Clusters"])
        for num in range(num_params):
            avg_sep_ratio[num], max_sep_ratio[num], avg_max_radius[num], overall_max_radius[num], avg_num_diff_clusters[num], max_num_diff_clusters[num] = PreservationSeparationFair(dist_matrix, dist_matrix_modded,
                                                                                                                                                                                      centers, radii, center_assignments, scr_radius, scale_params[
                                                                                                                                                                                          num],
                                                                                                                                                                                      num_iters, verbose=verbose)

            if verbose:
                print("""Scr with fair clustering and scale param {} yields optimal average maximum radius ratio {}, 
                    average separation ratio {}, max separation ratio {}, average number of different clusters {}, 
                    overall max number of different clusters {}, average max radius {}, and overall max radius {}""".format(scale_params[num],
                                                                                                                            avg_max_radius[num]/opt_radius, avg_sep_ratio[num], max_sep_ratio[num], avg_num_diff_clusters[num], max_num_diff_clusters[num], avg_max_radius[num], overall_max_radius[num]))

            f_writer.writerow(['Scale Param ' + str(num+1), scale_params[num], avg_sep_ratio[num], max_sep_ratio[num], avg_max_radius[num]/opt_radius,
                               overall_max_radius[num]/opt_radius, avg_max_radius[num], overall_max_radius[num], avg_num_diff_clusters[num], max_num_diff_clusters[num]])


if __name__ == "__main__":

    # Command line option parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', dest="input_file",
                        type=int, default="1", help="pmed file number")
    parser.add_argument('--num_iters', dest="num_iters",
                        type=int, default=1000, help="number of iterations")
    parser.add_argument('--scale_file', dest="scale_file", type=str,
                        default=None, help="file to manually enter scale parameters")
    parser.add_argument('--all', dest="all",
                        action="store_true", help="do all 40 pmed files")
    parser.add_argument('--out_folder', dest="out_folder", type=str,
                        default="AllPmedOutputs", help="where the output will store")
    parser.add_argument('--output_file', dest="output_file",
                        type=str, default=None, help="manually name output_file")
    parser.add_argument('-v', '--verbose', dest="verbose",
                        action='store_true', help="print stuff to stdout")
    parser.set_defaults(verbose=False)
    parameters = vars(parser.parse_args())

    scale_file = parameters["scale_file"]
    verbose = parameters['verbose']
    all_files = parameters['all']
    output_folder = parameters['out_folder']
    input_file_number = parameters['input_file']
    output_file = parameters['output_file']

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if (scale_file):
        with open(scale_file, "r") as f:
            scale = csv.reader(f, delimiter=",")
            for row in scale:
                input_scales = row
                input_scales = (float(x) for x in input_scales)

    if all_files:
        for input_file_number_ind in range(1, 41):
            base_filename = "pmed" + str(input_file_number_ind)
            input_file = "data/BeasleyData/" + base_filename + ".txt"
            if (not parameters['output_file']):
                output_file = output_folder + "/" + base_filename + ".txt"
                output_CSV = output_folder + "/" + base_filename + ".csv"

            else:
                output_file = output_folder + "/" + output_file + ".txt"
                output_CSV = output_folder + "/" + output_file + ".csv"

            num_iters = parameters['num_iters']
            ConductSimulation(num_iters, input_file_number_ind,
                              output_CSV, output_folder, verbose)
    else:
        base_filename = "pmed" + str(input_file_number)
        input_file = "data/BeasleyData/" + base_filename + ".txt"
        if (not parameters['output_file']):
            output_file = output_folder + "/" + base_filename + ".txt"
            output_CSV = output_folder + "/" + base_filename + ".csv"

        else:
            output_file = output_folder + "/" + output_file + ".txt"
            output_CSV = output_folder + "/" + output_file + ".csv"

        num_iters = parameters['num_iters']
        ConductSimulation(num_iters, input_file_number,
                          output_CSV, output_folder, verbose)
