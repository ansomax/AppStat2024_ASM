# --------------------------------------------------------------------------------------------------------
# Script for reading data for Problem 4.1 in Applied Statistics 2024-25 Problem Set:
# --------------------------------------------------------------------------------------------------------
#
# The data file contains a header and 63 entries in three columns:
#  * Year
#  * Population of India that year (PopIndia)
#  * Population of China that year (PopChina)
#
#   Author: Troels Petersen (Niels Bohr Institute, petersen@nbi.dk)
#   Date:   7th of November 2024 (latest version)
#
# --------------------------------------------------------------------------------------------------------

import numpy as np

data = np.genfromtxt('data_LargestPopulation.csv', delimiter = ',', skip_header=1)
print(data.shape)

year      = data[:,0]
PopIndia  = data[:,1]
PopChina  = data[:,2]

print(year)
print(PopIndia)
print(PopChina)
