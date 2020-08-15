import numpy as np
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse


if __name__ == "__main__":

    # Command line option parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_folder', dest="output_folder", default="AllAdultOutputs",
                        help="folder where the results of the simulations on the adult dataset were stored")
    parser.add_argument('--k_start', dest="k_start", type=int, default=2,
                        help="starting value for number of clusters that the simulations were ran for (inclusive)")
    parser.add_argument('--k_end', dest="k_end", type=int, default=21,
                        help="ending value for number of clusters that the simulations were ran for (exclusive)")
    parameters = vars(parser.parse_args())

    k_start = parameters['k_start']
    k_end = parameters['k_end']
    output_folder = parameters["output_folder"]

    scr_max_radius = []
    fair_max_radius = [[], [], []]

    scr_sep_ratio = []
    fair_sep_ratio = [[], [], []]

    scr_max_clusters = []
    fair_max_clusters = [[], [], []]

    base_filename = f"{output_folder}/adult_dataset_"
    num_clusters = list(range(k_start, k_end))
    for k in num_clusters:
        with open(f"{base_filename}{k}.csv", newline='') as csvfile:
            reader = csv.reader(csvfile)
            ind = 0
            for row in reader:
                if row[0] == "Unfair Clustering Method":
                    scr_sep_ratio.append(float(row[3]))
                    scr_max_radius.append(float(row[4]))
                    scr_max_clusters.append(float(row[6]))

                if row[0].startswith("Scale"):
                    fair_sep_ratio[ind].append(float(row[3]))
                    fair_max_radius[ind].append(float(row[4]))
                    fair_max_clusters[ind].append(float(row[7]))
                    ind = ((ind+1) % 3)

    scr_max_radius = np.array(scr_max_radius)
    scr_sep_ratio = np.array(scr_sep_ratio)
    scr_max_clusters = np.array(scr_max_clusters)

    fair_max_radius = np.array(fair_max_radius)
    fair_sep_ratio = np.array(fair_sep_ratio)
    fair_max_clusters = np.array(fair_max_clusters)

    # Plot of maximum radius against number of clusters
    fig, ax1 = plt.subplots()

    scr_radius_color = 'tab:red'
    fair_radius_color = ['#35cae8', '#156eeb', '#050bb5']
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Maximum Radius')
    ax1.plot(num_clusters, scr_max_radius,
             color=scr_radius_color, label='Scr Maximum Radius')

    ax1.plot(num_clusters, fair_max_radius[0], color=fair_radius_color[0],
             label=r"Fair Maximum Radius($\lambda = 16/R_{Scr}$)")
    ax1.plot(num_clusters, fair_max_radius[1], color=fair_radius_color[1],
             label=r"Fair Maximum Radius($\lambda = 4/R_{Scr}$)")
    ax1.plot(num_clusters, fair_max_radius[2], color=fair_radius_color[2],
             label=r"Fair Maximum Radius($\lambda = 1/R_{Scr}$)")

    ax1.tick_params(axis='y')

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles1, labels1 = ax1.get_legend_handles_labels()
    lgd1 = ax1.legend(handles1, labels1, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15))

    plt.title("Maximum Radius vs. Number of Clusters")

    plt.savefig(f"{output_folder}/adult_dataset_max_radius_vs_num_clusters.png",
                bbox_extra_artists=(lgd1,), bbox_inches='tight')
    plt.clf()

    # Plot of maximum pairwise separation ratio against number of clusters
    fig, ax2 = plt.subplots()

    ax2.set_xlabel('Number of Clusters')
    scr_separation_color = 'tab:red'
    fair_separation_color = fair_radius_color
    ax2.set_ylabel('Maximum Pairwise Separation Ratio')
    ax2.plot(num_clusters, scr_sep_ratio, color=scr_radius_color,
             label="Scr Maximum Separation Ratio")

    ax2.plot(num_clusters, fair_sep_ratio[0], color=fair_separation_color[0],
             label=r"Fair Maximum Separation Ratio($\lambda = 16/R_{Scr}$)")
    ax2.plot(num_clusters, fair_sep_ratio[1], color=fair_separation_color[1],
             label=r"Fair Maximum Separation Ratio($\lambda = 4/R_{Scr}$)")
    ax2.plot(num_clusters, fair_sep_ratio[2], color=fair_separation_color[2],
             label=r"Fair Maximum Separation Ratio($\lambda = 1/R_{Scr}$)")

    ax2.tick_params(axis='y')

    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    handles2, labels2 = ax2.get_legend_handles_labels()
    lgd2 = ax2.legend(handles2, labels2, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15))

    plt.title("Maximum Separation Ratio vs. Number of Clusters")

    plt.savefig(f"{output_folder}/adult_dataset_max_separation_ratio_vs_num_clusters.png",
                bbox_extra_artists=(lgd2,), bbox_inches='tight')
    plt.clf()

    # Plot of maximum number of different clusters against number of clusters
    fig, ax3 = plt.subplots()

    ax3.set_xlabel('Number of Clusters')
    scr_different_clusters_color = 'tab:red'
    fair_different_clusters_color = fair_radius_color
    ax3.set_ylabel('Maximum Number of Different Clusters')
    ax3.plot(num_clusters, scr_max_clusters, color=scr_radius_color,
             label="Scr Maximum Number of Different Clusters")

    ax3.plot(num_clusters, fair_max_clusters[0], color=fair_different_clusters_color[0],
             label=r"Fair Maximum Number of Different Clusters($\lambda = 16/R_{Scr}$)")
    ax3.plot(num_clusters, fair_max_clusters[1], color=fair_different_clusters_color[1],
             label=r"Fair Maximum Number of Different Clusters($\lambda = 4/R_{Scr}$)")
    ax3.plot(num_clusters, fair_max_clusters[2], color=fair_different_clusters_color[2],
             label=r"Fair Maximum Number of Different Clusters($\lambda = 1/R_{Scr}$)")

    ax3.tick_params(axis='y')

    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    handles3, labels3 = ax3.get_legend_handles_labels()
    lgd3 = ax3.legend(handles3, labels3, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15))

    plt.title("Maximum Number of Different Clusters vs. Number of Clusters")

    plt.savefig(f"{output_folder}/adult_dataset_max_different_clusters_vs_num_clusters.png",
                bbox_extra_artists=(lgd3,), bbox_inches='tight')
