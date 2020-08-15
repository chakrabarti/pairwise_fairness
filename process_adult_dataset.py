import numpy as np
import random
from scipy.spatial import distance
import sklearn.preprocessing as preprocessing


def CreateAdultDataset(num_samples: int = 1000, normalize: bool = False):
    # Generate seeded random sample of the adult dataset
    random.seed(1)
    lst = []
    count = 0

    # Use reservoir sampling to sample num_samples
    with open("data/adult.data") as f:
        for k, line in enumerate(f):
            if count < num_samples:
                split_line = line.split(", ")
                # These indices correspond to the attributes of interest: age, education-num, and hours per week
                lst.append([float(split_line[0]), float(
                    split_line[4]), float(split_line[12])])
                count += 1
            else:
                i = random.randint(0, k)
                if i < num_samples:
                    split_line = line.split(", ")
                    lst[i] = [float(split_line[0]), float(
                        split_line[4]), float(split_line[12])]
    lst = np.array(lst)

    # Remove duplicates
    lst = np.unique(lst, axis=0)

    # Normalize points
    if normalize:
        scaler = preprocessing.StandardScaler()
        scaler.fit(lst)
        lst = scaler.transform(lst)

    # Return Euclidean distance matrix between datapoints
    return distance.cdist(lst, lst, 'euclidean')


"""
Our simulations use 1000 normalized samples and feed this binary data in 
directly (so that for different simulations we don't have to make the sample
repeatedly we make a binary file and reuse it)
"""
if __name__ == "__main__":
    dist_matrix = CreateAdultDataset(num_samples=1000, normalize=True)

    # Create numpy binary file of adult dataset
    with open('data/adult_dataset_subsampled_normalized.in', 'wb') as f:
        np.save(f, dist_matrix)
