# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:40:39 2023

This code has been translated using chatGPT - 3.5 from the original MATLAB scripts
provided as ancillary files to https://arxiv.org/abs/1203.2995 by Jason L. Williams.


The traslated files were then edited by George Jones as the translation, whilst correct in
majority, fall foul of some common errors that prevent the code from running.

Appended to the translated code is the calucaltion of the GOSPA error, with the code
for these calculations being provided in https://github.com/ewilthil/gospapy/tree/master/gospapy

Along with this GOSPA error calculation, some code has been added to plot the GOSPA error and its 
constituent parts (localisation, missed and false targets).

References:
    
    - J. Williams, Marginal multi-Bernoulli filters: RFS derivation of MHT, 
      JIPDA and association-based MeMBer, IEEE Transactions on Aerospace and 
      Electronic Systems, 2015
      
      
    - J. Williams and R. Lau, Approximate evaluation of marginal association
    probabilities with belief propagation, IEEE Transactions on Aerospace and 
    Electronic Systems , 2014
    
    
    - A. S. Rahmathullah, A. F. Garcia-Fernandez and L. Svensson, Generalized
      optimal sub-pattern assignment metric, 20th International Conference on
      Information Fusion, 2017.
    

@author: George Jones
"""

import numpy as np
import matplotlib.pyplot as plt
from groundtruth import gentruth
from predict import predict
from update import update
from lbp import lbp
from tomb import tomb
from momb import momb

from gospa import calculate_gospa

# Set algorithm
alg = 'TOMB'  # Use the TOMB algorithm
# alg = 'MOMB'  # Use the MOMB algorithm

# Set simulation parameters
Pd = 0.9  # Probability of detection
lfai = 2  # Expected number of false alarms per scan
numtruth = 6  # Number of targets
simcasenum = 2 # Simulation case 1 or 2
if simcasenum == 1:
    Pmid = 1e-6 * np.eye(4)
else:
    Pmid = 0.25 * np.eye(4)

# Generate truth data
model, measlog, xlog = gentruth(Pd, lfai, numtruth, Pmid, simcasenum)

# Initialize filter parameters
stateDimensions = model['xb'].shape[0]
n = 0 # estimated number of targets
r = np.zeros((n, 1)) # probability of existence for each multi_Bernoulli component
x = np.zeros((stateDimensions, n)) # estimated states
P = np.zeros((n, stateDimensions, stateDimensions)) # estimated covariances
lambdau = model['lambdau'] # PPP intensity for the birth process (birth intensity)
xu = model['xb']
Pu = model['Pb']

# placeholder lists for filter estimates and the gospa error breakdown
estimates = []
all_gospa = []
all_loc = []
all_miss = []
all_fal = []


# Loop through time
numTime = len(measlog)
plt.figure(1)
for t in range(numTime):
    # Predict step
    r, x, P, lambdau, xu, Pu = predict(r, x, P, lambdau, xu, Pu, model)

    # Update step
    lambdau, xu, Pu, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = update(lambdau, xu, Pu, r, x, P, measlog[t], model)

    # Use Loopy Belief Propagation (LBP) to estimate association probabilities
    pupd, pnew = lbp(wupd, wnew)

    # Form new multi-Bernoulli components using either TOMB or MOMB algorithm
    if alg == 'TOMB':
        r, x, P = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
        ss = r > model['existThresh']
        n = np.sum(ss)
        
        
    elif alg == 'MOMB':
        r, x, P = momb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
        ss = np.zeros_like(r, dtype=bool)
        if len(r) > 0:
            pcard = np.prod(1 - r) * np.poly(-r / (1 - r))
            n = np.argmax(pcard)
            o = np.argsort(-r)
            n = n - 1
            ss[o[:n]] = True
    else:
        raise ValueError(f'Unknown algorithm: {alg}')
    
    x_ss = x[:,ss]
    estimates.append(x_ss)
    
    # Display the result
    plt.clf()
    plt.plot(xlog[t][0], xlog[t][2], 'k^', label="groundtruth")
    plt.plot(x[0, ss], x[2, ss], 'b*', label = "estimates")
    plt.plot(measlog[t][0], measlog[t][1], 'mx', label="measurements")
    plt.axis([-100, 100, -100, 100])
    plt.gca().set_aspect('equal', adjustable='box')
    if t <= 20:  # Slow to draw -- only draw for the first 20 time steps
        if n == 0:
            plt.legend(loc='upper right')
        else:
            plt.legend(loc='upper right')
            
    plt.title(f'{alg}; Number of MB components: {len(r)}; Birth PPP mixture components: {len(lambdau)}\nEstimated number of targets: {n}')
    plt.draw()
    plt.pause(0.1)

    
    # calucating the GOSPA error
    c = 10 # maximium localiastion error
    p = 2 # penalisation term, p=2
    
    # converting arrays to lists for ease of understanding when passing to the calculate_gospa function
    states_list = []
    for i in range(xlog[t].shape[1]):
        states_list.append(xlog[t][:,[i]])
    
    estimates_list = []
    for i in range(estimates[t].shape[1]):
        estimates_list.append(estimates[t][:,[i]])
        
    # states_list is groundtruth, estimates_list is filter estimates
    gospa, target_to_track_assigments, gospa_localization, gospa_missed, gospa_false = calculate_gospa(states_list, estimates_list, c, p)
    
    # gospa has already been **(1/p) in the function defenition but the localisation, missed and false have not
    gospa = gospa**2
    
    all_gospa.append(np.sqrt(gospa))
    all_loc.append(np.sqrt(gospa_localization))
    all_miss.append(np.sqrt(gospa_missed))
    all_fal.append(np.sqrt(gospa_false))
    
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
ax1.plot(range(201), all_gospa)
ax2.plot(range(201), all_loc)
ax3.plot(range(201), all_miss)
ax4.plot(range(201), all_fal)

ax1.set_title('GOSPA')
ax2.set_title('Localisation error')
ax3.set_title('Missed error')
ax4.set_title('False error')
ax1.set_ylim(-5, 20)
ax2.set_ylim(-5, 20)
ax3.set_ylim(-5, 20)
ax4.set_ylim(-5, 20)
plt.tight_layout()

