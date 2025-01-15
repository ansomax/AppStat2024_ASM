#Oversigt hvilke funktioner bliver nødvendige:

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.special import gammaln
from iminuit import Minuit
from scipy.stats import cost

def gauss(x, mu, sigma, Ngauss, binwidth) :
    """Gaussian function, from "unnormalizing" the PDF"""
    return Ngauss * binwidth * 1.0 / np.sqrt(2*np.pi) / sigma * np.exp( -0.5 * (x-mu)**2 / sigma**2)

def exp_pdf(x, tau, binwidth, Nexp):
    """Exponential function with lifetime tau, from "unnormalizing" the PDF"""
    return Nexp * binwidth * 1.0 / tau * np.exp(-x/tau)

def histogram(data, bin_count, hist_range):

    counts, bin_edges = np.histogram(data, bins=bin_count, range=hist_range)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2


    x = bin_centers[counts>0]
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0])

    return x, y, sy

#Tænker en fitting funktion for: binomial, poisson, gauss, som giver chi2, p-værdi, og plotter fit og data

def binomial_distribution(x, n, p, N_exp):
    ''' Binomial distribution. Takes n and p as input and returns the binomial distribution'''
    log_binomial_coefficient = gammaln(n + 1) - (gammaln(x + 1) + gammaln(n - x + 1))
    
    binomial_pmf = np.exp(log_binomial_coefficient(n, x) + x * np.log(p) + (n - x) * np.log(1 - p))

    func_binomial_2 = N_exp * binomial_pmf(x, n, p)

    return func_binomial_2

def poisson_distribution(mu, n):
    

    poisson_distribution = sc.stats.poisson.pmf(np.arange(n+1), mu)
    poisson_sigma = np.sqrt(mu) # For a Poisson process, the variance is mu

    return poisson_distribution, poisson_sigma

def weighted_mean(vals,sigs):
    values = np.array(vals)
    uncertainties = np.array(sigs)

    # Check if lengths match
    if len(values) != len(uncertainties):
        raise ValueError("Values and uncertainties must have the same length.")
    # Check for zero uncertainties to avoid division errors
    if np.any(uncertainties <= 0):
        raise ValueError("Uncertainties must be positive and non-zero.")

    # Calculate the weights
    weights = 1 / (uncertainties ** 2)
    # Calculate the weighted mean
    weighted_mean = np.sum(weights * values) / np.sum(weights)
    # Calculate the uncertainty of the weighted mean
    mean_uncertainty = np.sqrt(1 / np.sum(weights))
    
    
    return weighted_mean, mean_uncertainty

#Tænker en funktion som tager to datasæt og returnerer z-værdi og p-værdi for om de er ens


#Og en der gør det samme for t-test (low statistics)
#one-sample, two-sample, paired



#Måske også en fisher exact, bare så jeg ikke glemmer hvordan den er 

#En fisher discriminant funktion

def two_sample_sep_z_calc(x,y):
    '''Calculates the separation between two datasets x and y. Does the same as 
    a z-test. Can give you z-value for a two-sample z-test, or it can work as a separation
    measure for two datasets. Returns both sep/z and p-value.'''

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    x_std = np.std(x, ddof = 1)
    y_std = np.std(y, ddof = 1)
    sep = np.abs((x_mean - y_mean)) / np.sqrt(x_std**2 + y_std**2)

    p_mean = 2.0*sc.stats.norm.cdf(-np.abs(z), loc=0, scale=1)

    return sep, p_mean

#ROC curve
def calc_ROC(hist1, hist2):

    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    y_sig, x_sig_edges, _ = hist1 
    y_bkg, x_bkg_edges, _ = hist2
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges):
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = np.zeros_like(y_sig) # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig) # False positive rate ()
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers): 
            
            # The cut mask
            cut = (x_centers < x)
            
            # True positive
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
            TPR[i] = TP / (TP + FN)                    # True positive rate
            
            # True negative
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives
            FPR[i] = FP / (FP + TN)                     # False positive rate            
            
        return FPR, TPR
    
    else:
        AssertionError("Signal and Background histograms have different bins and/or ranges")
#Kolmogorov-Smirnoff test også, helt sikkert

def kolmogorov_smirnoff(x, y, alternative='two-sided'):
    '''Compatibility between function and sample or between two independent samples.
    Remember you must know the full distribution of the data to use this test.
    this statistic is calculated under the null hypothesis that the sample is drawn from the reference distribution 
    (in the function/sample case) or that the samples are drawn from the same distribution (in the two-sample case)'''
    ks, p = sc.stats.ks_2samp(x, y, alternative=alternative)
    return ks, p




#Binned likelihood (using Poisson) for histograms. 
#MAximum likelihood fit funktion er bedre for small statistics
#Unbinned likelihood (using PDF) for single values.
#En unbinned-likelihood fit funktion
# En binned likelihood fit funktion/chi2 fit funktion


#Noget til pplotting af data, med errorbars og fit

#Noget til histogrammer (1d og 2d) med fit og errorbars
