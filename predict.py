# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:44:42 2023

This code has been translated using chatGPT - 3.5 from the original MATLAB scripts
provided as ancillary files to https://arxiv.org/abs/1203.2995 by Jason L. Williams.

@author: sggjone5
"""

import numpy as np

def predict(r, x, P, lambdau, xu, Pu, model):
    # Get multi-Bernoulli prediction parameters from the model
    F = model['F']  # Transition matrix for the state
    Q = model['Q']  # Process noise covariance matrix
    Ps = model['Ps']  # Survival probability

    # Get birth parameters from the model
    lambdab = np.array([model['lambdab']])  # Birth intensity
    lambdau = np.array([lambdau])  # Existing track intensity
    nb = len(lambdab)  # Number of birth components
    xb = model['xb']  # Birth component state
    Pb = model['Pb']  # Birth component covariance matrix
    lambdab_threshold = 1e-4  # A threshold for low birth intensity components

    # Determine the number of measurements and existing tracks
    n = len(r)  # Number of measurements
    nu = len(np.array([lambdau]))  # Number of existing tracks (treated as a scalar)

    # Predict existing tracks
    for i in range(n):
        r[i] = Ps * r[i]  # Update existence probability for each track
        x[:, i] = np.dot(F, x[:, i])  # Predict the state for each existing track
        P[i, :, :] = np.dot(F, np.dot(P[i, :, :], F.T)) + Q  # Predict the covariance for each track

    # Predict existing PPP (Poisson Point Process) intensity
    for k in range(nu):
        lambdau[k] = Ps * lambdau[k]  # Update the intensity for each PPP component
        xu[:, k] = np.dot(F, xu[:, k])  # Predict the state for each PPP component
        Pu[k, :, :] = np.dot(F, np.dot(Pu[k, :, :], F.T)) + Q  # Predict the covariance for each PPP component

    # Incorporate birth intensity into the PPP

    # Allocate memory for new birth components
    lambdau = np.append(lambdau, np.zeros(nb))  # Append zeros for new birth components
    xu = np.column_stack((xu, np.zeros((len(xu), nb))))  # Add columns for new birth components to state matrix
    Pu = np.concatenate((Pu, np.zeros((nb, len(xu), len(xu)))), axis=0)  # Add new birth component covariances

    for k in range(nb):
        lambdau[nu + k] = lambdab[k]  # Set intensity for each new birth component
        xu[:, nu + k] = xb[:, k]  # Set the state for each new birth component
        Pu[nu + k, :, :] = Pb[k, :, :]  # Set the covariance for each new birth component

    # Not shown in the paper -- truncate low-weight components
    ss = lambdau > lambdab_threshold  # Identify birth components with sufficient intensity
    lambdau = lambdau[ss]  # Remove birth components with low intensity
    xu = xu[:, ss]  # Remove the corresponding state components
    Pu = Pu[ss, :, :]  # Remove the corresponding covariance components

    return r, x, P, lambdau, xu, Pu
