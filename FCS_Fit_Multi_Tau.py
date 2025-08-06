import numpy as np
import csv
import multipletau  # Install this package
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math


# Parameters of the fixed tau to use: 

Green_td = 0.0001715608189753372
Red_td = 0.00013882742882534565

k_green = -11.692949698263954
w_green = 3.735508891679675e-07
k_red = 1876731.7235220107
w_red = 4.149518739269024e-07


def runall():
    # Use global variables for green and red data arrays
    global green, red
    number_of_files = 1
    filename = "LUVs_TX_0.1_1in100"
    toloadpath = r"/Users/Mathew/Documents/Current analysis/FCS/FCS_LUVwithAF488_AF647/"

    green, red = [], []  # Initialize lists to store data from CSV files

    # Function to load data from a file
    def load_data(file):
        with open(file) as csvDataFile:
            csvReader = csv.reader(csvDataFile, delimiter='\t')
            for row in csvReader:
                green.append(row[0])
                red.append(row[1])

    # Load initial file
    load_data(toloadpath + filename)

    # Load additional files
    for i in range(1, number_of_files):
        suffix = f"_{i + 1:02}"  # Format suffix with zero padding
        load_data(toloadpath + filename + suffix)

    # Perform autocorrelation and crosscorrelation
    autocorrelate()
    crosscorrelate()

def autocorrelate():
    global new_c, new_d

    # Convert 'green' list to numpy array for processing
    x_green = np.array(green, dtype=float)
    # Perform autocorrelation on green data
    c = multipletau.autocorrelate(x_green, m=16, normalize=True, deltat=1e-5)
    # Remove the first row (t=0)
    new_c = np.delete(c, 0, axis=0)

    # Convert 'red' list to numpy array for processing
    x_red = np.array(red, dtype=float)
    # Perform autocorrelation on red data
    d = multipletau.autocorrelate(x_red, m=16, normalize=True, deltat=1e-5)
    # Remove the first row (t=0)
    new_d = np.delete(d, 0, axis=0)

def crosscorrelate():
    global new_e

    # Convert 'green' and 'red' lists to numpy arrays
    x = np.array(green, dtype=float)
    y = np.array(red, dtype=float)
    # Perform cross-correlation between green and red data
    e = multipletau.correlate(x, y, m=16, normalize=True, deltat=1e-5)
    # Remove the first row (t=0)
    new_e = np.delete(e, 0, axis=0)

def fitmultgreen(x,n,f1,td2):
    # Green multi fitting function
    return (1/n) * (f1*(((1 + x/Green_td) ** -1) * ((1 + x/(k_green ** 2 * Green_td)) ** -0.5))+(1-f1)*(((1 + x/td2) ** -1) * ((1 + x/(k_green ** 2 * td2)) ** -0.5)))

def fitgreen():
    # Prepare data for fitting
    xdata, ydata = new_c[:, 0], new_c[:, 1]
    
    guesses = [20,0.5,6e-4]  # Initial parameter guesses for curve fitting
    bounds = ([0,0,0], [np.inf, 1, np.inf])
    
    
    (n_,f1_,td2_), _ = opt.curve_fit(fitmultgreen, xdata, ydata, p0=guesses,bounds=bounds)

    y_fit = fitmultgreen(xdata, n_,f1_,td2_)  # Calculate fitted curve

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_title('FCS Curve')
    ax1.plot(xdata, ydata, "--", color="grey", label="correlate (numpy)")
    ax1.plot(xdata, y_fit, '-', color='green')
    ax1.set_xscale('log')
    ax1.set_xlim(1e-5, 10)
    ax1.set_ylim(0, max(new_c[:, 1]))
    ax1.set_xlabel("Time lag (s)")
    ax1.set_ylabel("Autocorrelation")

    #Print results of fitting
    
    print(f"Green_N = {n_}")
    
    print(f"Green_td1 = {Green_td}")
    print(f"Green_td2 = {td2_}")
    
    print(f"Green_f1 = {f1_}")
    print(f"Green_f2 = {1-f1_}")
    
def fitmultred(x,n,f1,td2):
    # Red multi fitting function
    return (1/n) * (f1*(((1 + x/Red_td) ** -1) * ((1 + x/(k_red ** 2 * Red_td)) ** -0.5))+(1-f1)*(((1 + x/td2) ** -1) * ((1 + x/(k_red ** 2 * td2)) ** -0.5)))
 


def fitred():
    # Prepare data for fitting
    xdata, ydata = new_c[:, 0], new_c[:, 1]
    
    guesses = [20,0.5,6e-4]  # Initial parameter guesses for curve fitting
    bounds = ([0,0,0], [np.inf, 1, np.inf])
    
    
    (n_,f1_,td2_), _ = opt.curve_fit(fitmultred, xdata, ydata, p0=guesses,bounds=bounds)

    y_fit = fitmultred(xdata, n_,f1_,td2_)  # Calculate fitted curve

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_title('FCS Curve')
    ax1.plot(xdata, ydata, "--", color="grey", label="correlate (numpy)")
    ax1.plot(xdata, y_fit, '-', color='red')
    ax1.set_xscale('log')
    ax1.set_xlim(1e-5, 10)
    ax1.set_ylim(0, max(new_c[:, 1]))
    ax1.set_xlabel("Time lag (s)")
    ax1.set_ylabel("Autocorrelation")

    #Print results of fitting
    
    print(f"Red_N = {n_}")
    
    print(f"Red_td1 = {Red_td}")
    print(f"Red_td2 = {td2_}")
    
    print(f"Red_f1 = {f1_}")
    print(f"Red_f2 = {1-f1_}")

# Run all functions
runall()
fitgreen()
fitred()

