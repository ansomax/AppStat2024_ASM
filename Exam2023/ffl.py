
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import sys
import scipy as sc
from scipy.optimize import curve_fit

sys.path.append('../External_Functions')

from ExternalFunctions import Chi2Regression, UnbinnedLH, BinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax 

##### STANDARD PLOT FUNCTIONS #####

def gauss(x, mu, sigma, N0=1):
    ''' Gaussian function. Takes x, mu and sigma as input and returns the Gaussian function.
        For standard normalization set N0 = 1'''
    return N0/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))

def expontential(x, tau, N0):
    ''' Exponential function. Takes x, tau and N0 as input and returns the exponential function.
        For standard normalization set tau = 1'''
    return N0/tau * np.exp(-x/tau)



# Histogramming and masking for fitting
def hist_for_fit(x, N_bins, hist_range, density_hist=False):
    ''' Takes x data and creates a histogram with the given number of bins and range.
        Returns the masked bin centers, counts and error on counts, ready for fitting'''
    counts, bin_edges = np.histogram(x, bins=N_bins, range=hist_range, density=density_hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    err_y = np.sqrt(counts)
    
    mask = counts > 0
    return bin_centers[mask], counts[mask], err_y[mask]


def mean_test(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    x_sdom = np.std(x, ddof = 1)/np.sqrt(len(x))
    y_sdom = np.std(y, ddof = 1)/np.sqrt(len(y))
    z = np.abs((x_mean - y_mean)) / np.sqrt(x_sdom**2 + y_sdom**2)
    
    p_mean = 2.0*sc.stats.norm.cdf(-np.abs(z), loc=0, scale=1)

    return z, p_mean
        


# Accept / reject
def acc_rej(N_points, func, arg, x_range, y_rang):
    x_dist = np.zeros(N_points)
    N_try = 0

    for i in range(N_points):
        while True:
            N_try += 1
            x = np.random.uniform(x_range[0], x_range[1]) 
            y = np.random.uniform(y_rang[0], y_rang[1])
            if (y < func(x, *arg)) :   # If the (x,y)-point fulfills the accept condition...
                x_dist[i] = x
                break  
            
    return x_dist, N_try

# Mixed method

def mix_method(N_points, func, arg, box_func, box_func_inv):
    x_dist = np.zeros(N_points)
    y_dist = np.zeros(N_points)
    N_try = 0

    for i in range(N_points):
        while True:
            N_try += 1
            x = box_func_inv(np.random.uniform(size=1))  
            y = np.random.uniform(0, box_func(x))        
            if (y < func(x, *arg)) :   # If the (x,y)-point fulfills the accept condition...
                break      
        x_dist[i] = x                   # ...then break and accept the x-value
        y_dist[i] = y                   # ...and the y-value
        
    return x_dist, y_dist, N_try



# Fitting and calc. chi2 with minuit function
def fit_with_chi2(f, x, y, sy, func_par):
    Chi2 = Chi2Regression(f, x, y, sy)
    minuit = Minuit(Chi2, *func_par)
    minuit.errordef = 1.0
    minuit.migrad();

    par = minuit.values[:]
    err = minuit.errors[:]
    par_names = list(minuit.parameters)
    chi2 = minuit.fval
    Ndof = len(x) - len(func_par)
    prob = sc.stats.chi2.sf(chi2, Ndof)


    # Creating dictionary with every parameter and its error
    par_dict = { par_names[i] : [par[i], err[i]] for i in range(len(par_names)) }

    # adding to dictionary chi2, Ndof and prob
    par_dict['chi2'] = chi2
    par_dict['Ndof'] = Ndof
    par_dict['p_val'] = prob

    return minuit, Ndof, prob, par_dict


def sep_calc(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    x_std = np.std(x, ddof = 1)
    y_std = np.std(y, ddof = 1)
    sep = np.abs((x_mean - y_mean)) / np.sqrt(x_std**2 + y_std**2)
    return sep


def easy_fisher(data1, data2):
    mean_1 = np.mean(data1, axis=0)
    mean_2 = np.mean(data2, axis=0)

    cov_1 = np.cov(data1.T)
    cov_2 = np.cov(data2.T)

    cov_tot = cov_1 + cov_2
    cov_tot_inv = np.linalg.inv(cov_tot)

    wf = np.matmul(cov_tot_inv, (mean_1 - mean_2))
    return data1.dot(wf), data2.dot(wf), wf


def calc_ROC(hist1, hist2) :
    ''' Used for contructing ROC curves. Takes two histograms as input and returns the False Positive Rate (FPR) and the True Positive Rate (TPR).
        IMPORTANT!!! Make sure to bin the histograms with the same binning and range. Look at 4_1 for guidance.'''
    
    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    y_sig, x_sig_edges, _ = hist1 
    y_bkg, x_bkg_edges, _ = hist2
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges) :
        
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
   

def UBLH_minuit(func, x, start_val):
    ''' A function for doing the initial Unbinned Likelihood fit. Takes a function, x values and starting values as input. 
        Returns the minuit fit and a dictionary with the fit values and errors. '''
    UBLH = UnbinnedLH(func, x, extended=True)
    minuit_fit_ublh = Minuit(UBLH, *start_val)
    minuit_fit_ublh.errordef = 0.5
    minuit_fit_ublh.migrad()
    

    fit_values_ublh = minuit_fit_ublh.values
    fit_errors_ublh = minuit_fit_ublh.errors
    fit_names_ublh = minuit_fit_ublh.parameters

    # Create dictionary with values and errors from fit with a loop
    fit_dict_ublh = { fit_names_ublh[i] : [fit_values_ublh[i], fit_errors_ublh[i]] for i in range(len(fit_values_ublh))}

    # add UBLH value to dictionary
    fit_dict_ublh['UBLH'] = minuit_fit_ublh.fval


    return minuit_fit_ublh, fit_dict_ublh


def UBLH_sim(N_exp, N_counts, x_from_rand, rand_param, func, start_param, N_bins_sim, Likelihood_fit_start_val):
    ''' Simulating the UBLH value for a given function. 
        Takes the number of experiments, original number of counts, and the functin for generating x values (often np.random.normal).
        Returns a list of UBLH values which can then be histogrammed along with the original UBLH value to give a statistical overview.'''

    ublh_list = []
    for _ in range(N_exp):
        x = x_from_rand(*rand_param, N_counts)
        fit_val, dict = UBLH_minuit(func, x, *start_param)
        UBHL_val = fit_val.fval
        ublh_list.append(UBHL_val)

    

    hist_range = [np.min(ublh_list), np.max(ublh_list)]
    counts, bin_edges = np.histogram(ublh_list, bins = N_bins_sim, range=hist_range, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    x = bin_centers[counts > 0]
    counts = counts[counts > 0]

    par, cov = curve_fit(func, x, counts, p0 = Likelihood_fit_start_val)
    x_plot = np.linspace(hist_range[0], hist_range[1], 1000)
    y_plot = func(x_plot, *par)

    fig, ax = plt.subplots()
    ax.hist(ublh_list, bins = N_bins_sim, histtype='stepfilled', alpha=0.7, range=hist_range, density=True)
    ax.plot(x_plot, y_plot, label='Fit')
    ax.axvline(np.mean(ublh_list), color = 'darkblue', ls='--', label = 'Mean UBLH value')
    ax.fill_between(x_plot, y_plot, 0, where = (x_plot > par[0] - par[1]) & (x_plot < par[0] + par[1]), alpha = 0.5, label = '1 sigma', color='blue')
    ax.set_xlabel('UBLH')
    ax.set_ylabel('Probability density')
    ax.set_title('UBLH distribution')
    ax.legend()
        
    return fig, ax, hist_range



def quick_hist(data, N_bins, hist_range):
    ''' Used for plotting a quick histogram of given data. 
        Takes the data, number of bins and range as input. Returns the figure and axis.'''
    fig, ax = plt.subplots()
    ax.hist(data, bins=N_bins, range=hist_range, histtype='stepfilled', alpha=0.7,  label='Data')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('Counts')
    ax.set_title('Quick histogram')
    plt.show()
    return fig, ax
