import numpy as np
import csv
import multipletau  # Install this package
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math

# Constants for diffusivity calculations
k_green = -11.692949698263954
w_green = 3.735508891679675e-07
k_red = 1876731.7235220107
w_red = 4.149518739269024e-07

def runall():
    # Use global variables for green and red data arrays
    global green, red
    number_of_files = 1
    filename = "Dye_20_nM"
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

def fungreen(x, n, td):
    # Green channel fitting function
    return (1/n) * ((1 + x/td) ** -1) * ((1 + x/(k_green ** 2 * td)) ** -0.5)

def fitgreen():
    # Prepare data for fitting
    xdata, ydata = new_c[:, 0], new_c[:, 1]
    guesses = [20, 6e-5]  # Initial parameter guesses for curve fitting
    (n_, td_), _ = opt.curve_fit(fungreen, xdata, ydata, p0=guesses)

    y_fit = fungreen(xdata, n_, td_)  # Calculate fitted curve

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

    # Print results of fitting - tau value

   

    return td_

def funred(x, n, td):
    # Red channel fitting function
    return (1/n) * ((1 + x/td) ** -1) * ((1 + x/(k_red ** 2 * td)) ** -0.5)

def fitred():
    # Prepare data for fitting
    xdata, ydata = new_d[:, 0], new_d[:, 1]
    guesses = [25, 5e-5]  # Initial parameter guesses for curve fitting
    (n_, td_), _ = opt.curve_fit(funred, xdata, ydata, p0=guesses)

    y_fit = funred(xdata, n_, td_)  # Calculate fitted curve

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_title('FCS Curve')
    ax1.plot(xdata, ydata, "--", color="grey", label="correlate (numpy)")
    ax1.plot(xdata, y_fit, '-', color='red')
    ax1.set_xscale('log')
    ax1.set_xlim(1e-5, 10)
    ax1.set_ylim(0, max(new_d[:, 1]))
    ax1.set_xlabel("Time lag (s)")
    ax1.set_ylabel("Autocorrelation")

    
    

    return td_



# Run all functions
runall()
green=fitgreen()
red=fitred()
print(f"Green_td = {green}")
print(f"Red_td = {red}")