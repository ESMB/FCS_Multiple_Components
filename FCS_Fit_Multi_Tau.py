import numpy as np
import csv
import multipletau  # Ensure this package is installed
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Parameters of the fixed tau to use
Green_td = 7.803075260171489e-05
Red_td = 0.0001068639930648766


# For the unknown, these are the upper and lower-bounds of td - useful for when proportion is very low,
# otherwise it will fit to a td close to the green and red provided tds. 

green_lower=0.001
green_upper=np.inf

red_lower=0.001
red_upper=np.inf

# The parameters below can be determined by fitting molecules that have a known diffusion coefficient.
k_green = -11.692949698263954
w_green = 3.735508891679675e-07
k_red = 1876731.7235220107
w_red = 4.149518739269024e-07

def runall():
    # Use global variables for green and red data arrays
    global green, red
    number_of_files = 1
    filename = "LUVs_TX_0.001_1in100"
    toloadpath = r"/Users/Mathew/Documents/Current analysis/FCS/FCS_LUVwithGFP_ATTO655DNA/"

    # Initialize lists to store data from CSV files
    green, red = [], []
    
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

    # Perform autocorrelation and cross-correlation
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

def fitmultgreen(x, n, f1, td2):
    # Green multi fitting function with parameters for optimization
    return (1/n) * (f1*(((1 + x/Green_td) ** -1) * ((1 + x/(k_green ** 2 * Green_td)) ** -0.5)) +
                    (1-f1)*(((1 + x/td2) ** -1) * ((1 + x/(k_green ** 2 * td2)) ** -0.5)))

def fitgreen():
    # Prepare data for fitting
    xdata, ydata = new_c[:, 0], new_c[:, 1]
    
    # Initial parameter guesses for curve fitting
    guesses = [20, 0.5, 6e-3]
    bounds = ([0, 0, green_lower], [np.inf, 1, green_upper])
    
    # Perform curve fitting
    (n_, f1_, td2_), _ = opt.curve_fit(fitmultgreen, xdata, ydata, p0=guesses, bounds=bounds)

    # Calculate fitted curve
    y_fit = fitmultgreen(xdata, n_, f1_, td2_)

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_title('FCS Curve')
    ax1.plot(xdata, ydata, "--", color="grey", label="correlate (numpy)")
    ax1.plot(xdata, y_fit, '-', color='green', label="fit")
    ax1.set_xscale('log')
    ax1.set_xlim(1e-5, 10)
    ax1.set_ylim(0, max(new_c[:, 1]))
    ax1.set_xlabel("Time lag (s)")
    ax1.set_ylabel("Autocorrelation")
    ax1.legend()

    # Print results of fitting
    print(f"Green_N = {n_}")
    print(f"Green_td1 = {Green_td}")
    print(f"Green_td2 = {td2_}")
    print(f"Green_f1 = {f1_}")
    print(f"Green_f2 = {1-f1_}")

def fitmultred(x, n, f1, td2):
    # Red multi fitting function with parameters for optimization
    return (1/n) * (f1*(((1 + x/Red_td) ** -1) * ((1 + x/(k_red ** 2 * Red_td)) ** -0.5)) +
                    (1-f1)*(((1 + x/td2) ** -1) * ((1 + x/(k_red ** 2 * td2)) ** -0.5)))

def fitred():
    # Prepare data for fitting
    xdata, ydata = new_c[:, 0], new_c[:, 1]
    
    # Initial parameter guesses for curve fitting
    guesses = [20, 0.5, 6e-3]
    bounds = ([0, 0, red_lower], [np.inf, 1, red_upper])
    
    # Perform curve fitting
    (n_, f1_, td2_), _ = opt.curve_fit(fitmultred, xdata, ydata, p0=guesses, bounds=bounds)

    # Calculate fitted curve
    y_fit = fitmultred(xdata, n_, f1_, td2_)

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_title('FCS Curve')
    ax1.plot(xdata, ydata, "--", color="grey", label="correlate (numpy)")
    ax1.plot(xdata, y_fit, '-', color='red', label="fit")
    ax1.set_xscale('log')
    ax1.set_xlim(1e-5, 10)
    ax1.set_ylim(0, max(new_c[:, 1]))
    ax1.set_xlabel("Time lag (s)")
    ax1.set_ylabel("Autocorrelation")
    ax1.legend()

    # Print results of fitting
    print(f"Red_N = {n_}")
    print(f"Red_td1 = {Red_td}")
    print(f"Red_td2 = {td2_}")
    print(f"Red_f1 = {f1_}")
    print(f"Red_f2 = {1-f1_}")

# Run all functions
runall()
fitgreen()
fitred()
