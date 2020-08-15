
# A Pairwise Fair and Community-preserving Approach to k-Center Clusterings
This is the codebase for our paper, "A Pairwise Fair and Community-preserving Approach to k-Center Clustering", which appeared in ICML 2020

## simulation_dump
contains the results of the runs (including the plots) that were used in the paper


## data

### BeasleyData
contains the pmed files

### adult.data
contains the adult dataset

### adult.names
contains the names corresponding to the columns of the adult dataset

### adult_dataset_subsampled.in
contains the subsampled adult data used in the paper's experiments

## clustering
contains all of the clustering algorithms
  

### gonzalez.py

contains the code for running the "unfair" Gonzalez and Gonzalez Plus algorithms

  

### scr.py

contains the code for running the "unfair" Scr algorithm

  

### fairalg.py

contains the code for running the fair algorithm described in the paper

## process_pmed.py

contains the code for processing BeasleyData to create distance matrices


## process_adult_dataset.py

subsamples from the adult dataset (adult.data) and creates a numpy binary file creating the normalized dataset


## simhelp.py

contains helper functions for running the simulations
  

## pmed_simulation.py

is the one of the main runscripts; runs simulations for the unfair algorithms and well as the fair algorithm (with Scr as the unfair algorithm used as input to the fair algorithm).


## adult_dataset_simulation.py

is the other main runscript; runs simulations for Scr as well as the fair algorithm on the subsampled adult dataset (using the generated numpy binary file) using a variable number of clusters (provided by argument)
  


## make_bulk_pmed_plots.py

is one of the main plotting scripts; assuming the simulations have been run for all 40 pmed files, will create a plot of the maximum average number of different clusters against the maximum radius to the optimal radius ratio and a plot of the maximum pairwise separation ratio against the maximum radius to the optimal radius ratio
  

## adult_dataset_plotting.py

creates 3 different plots based on the results of adult_dataset_simulation.py: maximum radius, maximum pairwise separation ratio, and maximum number of different clusters vs. the number of clusters
  
## requirements.txt
lists Python dependencies

## Running the code

1. Run "pip install -r requirements.txt"

2. Call *python pmed_simulation.py --all* (alternatively for *i* in [1,41) call *python pmed_simulation.py --input_file i*)

3. Call *python make_bulk_pmed_plots.py* to produce the two plots described above (they will be produced in the *AllPmedOutputs* folder)

4. For *num_clusters* in [2, 21), call *python adult_dataset_simulation.py -k num_clusters*

5. Call *python adult_dataset_plotting.py* to produce the three plots described above (they will be produced in the *AllAdultOutputs* folder).