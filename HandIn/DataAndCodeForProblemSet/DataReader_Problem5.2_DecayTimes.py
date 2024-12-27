# --------------------------------------------------------------------------------------------------------
# Script for reading data for Problem 5.2 in Applied Statistics 2024-25 Problem Set:
# --------------------------------------------------------------------------------------------------------
#
# The data file contains 1000 entries with measured decay times in seconds.
#
#   Author: Troels Petersen (Niels Bohr Institute, petersen@nbi.dk)
#   Date:   7th of November 2023 (latest version)
#
# --------------------------------------------------------------------------------------------------------

import numpy as np

data = np.genfromtxt('data_DecayTimes.csv')
print(data.shape)

# Print the first ten entries to get a feel for the data:
print(data[0:10])
