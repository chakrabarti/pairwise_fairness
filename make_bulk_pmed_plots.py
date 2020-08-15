import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
import argparse

if __name__ == "__main__":

    # Command line option parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_folder', dest="output_folder", default="AllPmedOutputs",
                        help="folder where the results of the simulations on the pmed dataset were stored")
    parser.add_argument('--start', dest="start", type=int, default=1,
                        help="starting value for the pmed file number for which the simulations were run for (inclusive)")
    parser.add_argument('--end', dest="end", type=int, default=41,
                        help="ending value for number of clusters that the simulations were run for (exclusive)")
    parameters = vars(parser.parse_args())

    start = parameters["start"]
    end = parameters["end"]

    # There are only 40 pmed files numbered 1 to 40 so if a larger number is provided accidentally, clip
    start = max(1, start)
    end = min(end, 41)
    target_folder = parameters["output_folder"]
    to_plot = list(range(start, end))

    radius_ratio = [[] for _ in range(3)]
    max_num_diff_clusters = [[] for _ in range(3)]
    max_sep_ratio = [[] for _ in range(3)]

    scr_zero = []
    scr_one = []
    scr_two = []
    gonz_one_zero = []
    gonz_one_one = []
    gonz_one_two = []
    gonz_plus_zero = []
    gonz_plus_one = []
    gonz_plus_two = []

    # All of the pmed files to be plotted
    for i in to_plot:
        instance_name = f"pmed{i}"
        input_name = f"{target_folder}/{instance_name}.csv"
        results_file = open(input_name, "r")

        ind = 0
        for line in results_file:
            parsed_line = line.split(",")
            if parsed_line[1] == "Scr":
                scr_zero.append(float(parsed_line[3]))
                scr_one.append(float(parsed_line[4]))
                scr_two.append(float(parsed_line[7]))
            elif parsed_line[1] == "Gonzalez":
                gonz_one_zero.append(float(parsed_line[3]))
                gonz_one_one.append(float(parsed_line[4]))
                gonz_one_two.append(float(parsed_line[7]))
            elif parsed_line[1] == "GonzalezPlus":
                gonz_plus_zero.append(float(parsed_line[3]))
                gonz_plus_one.append(float(parsed_line[4]))
                gonz_plus_two.append(float(parsed_line[7]))
            elif parsed_line[0].startswith("Scale Param"):
                param = int(parsed_line[0].split(" ")[2]) - 1
                if not (param % 2):
                    radius_ratio[ind].append(float(parsed_line[4]))
                    max_num_diff_clusters[ind].append(float(parsed_line[9]))
                    max_sep_ratio[ind].append(float(parsed_line[3]))
                    ind = (ind + 1) % 3

    # Plot of maximum average different clusters
    plt.figure(i)
    plt.rcParams.update({'font.size': 14})

    fair_colors = ['#35cae8', '#156eeb', '#050bb5']
    plt.plot(radius_ratio[0], max_num_diff_clusters[0], 'o',
             color=fair_colors[0], label=r"Fair ($\lambda = 16/R_{Scr}$)")
    plt.plot(radius_ratio[1], max_num_diff_clusters[1], 'o',
             color=fair_colors[1], label=r"Fair ($\lambda = 4/R_{Scr}$)")
    plt.plot(radius_ratio[2], max_num_diff_clusters[2], 'o',
             color=fair_colors[2], label=r"Fair ($\lambda = 1/R_{Scr}$)")

    plt.xlabel("Radius Ratio")
    plt.ylabel("Max Average Number of Clusters")
    plt.suptitle("Max Average Number of Clusters vs. Radius Ratio")
    plt.plot(scr_one, scr_two, 'Dr', label="Scr")
    plt.plot(gonz_one_one, gonz_one_two, '^m', label="Gonzalez")
    plt.plot(gonz_plus_one, gonz_plus_two, 'sy', label="GonzalezPlus")

    plt.yscale('log')
    plt.xscale('log')
    ax = plt.gca()

    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax.yaxis.set_minor_locator(plt.FixedLocator([2, 3, 4, 5, 6, 7, 8]))

    plt.legend()
    plt.legend(loc=1, prop={'size': 12})
    plt.savefig(f"{target_folder}/CommunityPlot{start}_{end-1}.png")

    # Plot of pairwise fairness
    plt.figure(i + 40)

    plt.rcParams.update({'font.size': 14})
    fair = plt.plot(radius_ratio[0], max_sep_ratio[0], 'o',
                    color='#35cae8', label=r"Fair ($\lambda = 16/R_{Scr}$)")
    plt.plot(radius_ratio[1], max_sep_ratio[1], 'o',
             color='#156eeb', label=r"Fair ($\lambda = 4/R_{Scr}$)")
    plt.plot(radius_ratio[2], max_sep_ratio[2], 'o',
             color='#050bb5', label=r"Fair ($\lambda = 1/R_{Scr}$)")
    plt.xlabel("Radius Ratio")
    plt.ylabel("Maximum Pairwise Separation Ratio")
    plt.suptitle("Maximum Pairwise Separation Ratio vs. Radius Ratio")
    plt.plot(scr_one, scr_zero, 'Dr', label="Scr")
    plt.plot(gonz_one_one, gonz_one_zero, '^m', label="Gonzalez")
    plt.plot(gonz_plus_one, gonz_plus_zero, 'sy', label="GonzalezPlus")
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    ax.xaxis.set_minor_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax.yaxis.set_major_locator(plt.FixedLocator([]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([1, 2, 4, 8, 20, 40, 70]))

    plt.legend()
    plt.legend(loc=1, prop={'size': 12})

    plt.savefig(f"{target_folder}/PairPlot{start}_{end-1}.png")
