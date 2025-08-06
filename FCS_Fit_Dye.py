import numpy as np
import pandas as pd
import csv
import multipletau               #Install this packaged
import matplotlib.pyplot as plt
import scipy.optimize as opt




########################################### Run all scripts ############################################
# Diffusion coefficients below
D_green=4.25e-10
D_red=4.25e-10

def runall():
    toloadsample = r"/Volumes/T7/Current_Analysis/20250205_FCS/Run 1/AF647/25nM/FCS"
    # This is the FCS file to load that has the data from the dye. This is to determine the beam waist. 
                                                 
    
    autocorrelate(toloadsample)                                                     # Autocorrelated the sample - comment out if you just want to look at the dye.
                                                                           

########################################### Autocorrelate ############################################
    
def autocorrelate(toload):
   global c
   global new_c
   global new_d
  


   green = []                                                                       # This is where the red and green data will be saved. Makes an array. Data exists as two columns of 10 us bursts (green and red)
   red = []               



   with open(toload) as csvDataFile:                                                # Opens the file as a CSV
       csvReader = csv.reader(csvDataFile,delimiter='\t')                           # Assigns the loaded CSV file to csvReader. 
       for row in csvReader:
           green.append(row[0])                                                     # For every row in in csvReader, the values are apended to green and red.         
           red.append(row[1])
     
        
   x=np.array(green,dtype=float)                                                    # Convert the csv columns to float values - Why does it not know?
   c=multipletau.autocorrelate(x,m=16, normalize=True,deltat=1e-5)                  # Correlate the data using the multipletau python. The deltat is the bin width. 
   new_c=np.delete(c, (0), axis=0)                                                  # This deletes the first row which is at t = 0. 
   
       
   
   x=np.array(red,dtype=float)                                                    # Convert the csv columns to float values - Why does it not know?
   d=multipletau.autocorrelate(x,m=16, normalize=True,deltat=1e-5)                  # Correlate the data using the multipletau python. The deltat is the bin width. 
   new_d=np.delete(d, (0), axis=0) 






def fungreen(x,n,k,w):                                                                   # This is for the fitting with the dye.      
    D=D_green                                                                      # Change for diffusion co-efficient of the dye. 
    return (1/n)*((1+(4*D*x)/(w**2))**(-1))*(1+(4*D*x)/(k**2*w**2))**(-0.5)         # 3D diffusion model with curve. 

def funred(x,n,k,w):                                                                   # This is for the fitting with the dye.      
    D=D_red                                                                      # Change for diffusion co-efficient of the dye. 
    return (1/n)*((1+(4*D*x)/(w**2))**(-1))*(1+(4*D*x)/(k**2*w**2))**(-0.5) 
########################################### Fit with known diffusion coefficient ############################################

def fitgreendye():
    xdata=new_c[:, 0]                       # Assigns the x-data.
    ydata=new_c[:,1]                        # Assigns the y-data.
    guesses= np.array([5,5,1e-6])        # Intitial guesses for n, k and w. 
    (n_, k_, w_), _ = opt.curve_fit(fungreen, xdata, ydata,guesses)
    params= opt.curve_fit(fungreen, xdata, ydata,guesses)
    
    y_fit = fungreen(xdata, n_, k_, w_)
    
    
   
# Plot the data:
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.set_window_title('FCS Curve')
    ax1 = fig.add_subplot(211)
    ax1.plot(xdata, ydata, "--k",
         color="grey", label="FCS Data")
    ax1.set_xlabel("Time lag (s)")
    ax1.set_ylabel("Autocorrelation")
    ax1.set_xscale('log')
    ax1.set_xlim(1e-5,10)
    ax1.set_ylim(0,(1/params[0][0]))
    ax1.plot(xdata, y_fit, '-',color="green")
    
    
    Green_k=params[0][1]
    Green_w=params[0][2]
    
    print(("Green_N = %r \r") %params[0][0])
    print(("Green_k = %r \r") %params[0][1])
    print(("Green_w = %r \r") %params[0][2])
    
    return Green_k,Green_w
    
def fitreddye():
    xdata=new_d[:, 0]                       # Assigns the x-data.
    ydata=new_d[:,1]                        # Assigns the y-data.
    guesses= np.array([5,5,1e-6])        # Intitial guesses for n, k and w. 
    (n_, k_, w_), _ = opt.curve_fit(funred, xdata, ydata,guesses)
    params= opt.curve_fit(funred, xdata, ydata,guesses)
    
    y_fit = funred(xdata, n_, k_, w_)
    
    
   
# Plot the data:
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.set_window_title('FCS Curve')
    ax1 = fig.add_subplot(211)
    ax1.plot(xdata, ydata, "--k",
         color="grey", label="FCS Data")
    ax1.set_xlabel("Time lag (s)")
    ax1.set_ylabel("Autocorrelation")
    ax1.set_xscale('log')
    ax1.set_xlim(1e-5,10)
    ax1.set_ylim(0,(1/params[0][0]))
    ax1.plot(xdata, y_fit, '-',color="red")
    
    Red_k=params[0][1]
    Red_w=params[0][2]
    
    print(("Red_N = %r \r") %params[0][0])
    print(("Red_k = %r \r") %params[0][1])
    print(("Red_w = %r \r") %params[0][2])
    
    return Red_k,Red_w
    

runall()
Green_k,Green_w=fitgreendye()
Red_k,Red_w=fitreddye()

print("-------For fit code-------")

print("k_green=%r \r"%Green_k)
print("w_green=%r \r"%Green_w)
print("k_red=%r \r"%Red_k)
print("w_red=%r \r"%Red_w)