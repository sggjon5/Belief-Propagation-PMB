# -*- coding: utf-8 -*-
"""
Created on Thu Oct 5 16:20:34 2023
@author: sggjone5
"""

import numpy as np

def gentruth(Pd, lfai, numtruth, Pmid, pmi):
    if pmi == 1:
        birthtime = np.zeros(numtruth)
    else:
        birthtime = 10 * np.arange(numtruth)

    # Initialize model parameters
    T = 1 # time step
    model = {}
    model['F'] = np.kron(np.eye(2), np.array([[1, T], [0, 1]])) # State transition matrix
    model['Q'] = 0.01 * np.kron(np.eye(2), np.array([[T**3/3, T**2/2], [T**2/2, T]])) # Process noise covariance matrix
    model['Qc'] = np.linalg.cholesky(model['Q']) # Cholesky decomposition of Q
    model['H'] = np.kron(np.eye(2), np.array([[1, 0]])) # Measurement matrix
    model['R'] = np.eye(2) # Measurement noise covariance matrix
    model['Rc'] = np.linalg.cholesky(model['R']) # Cholesky decomposition of R
    model['Pd'] = Pd # probability of detection
    model['Ps'] = 0.999 # Track survival probability
    model['existThresh'] = 0.8 # Track existence probability threshold

    # Initialize new target parameter structure
    model['xb'] = np.zeros((4, 1))
    model['Pb'] = np.expand_dims(np.diag([100, 1, 100, 1])**2, 0)
    model['lambdau'] = 10  # initially expect 10 targets present (regardless of the true number)
    volume = 200 * 200
    model['lambdab'] = 0.05  # expect one new target to arrive every 20 scans on average
    model['lfai'] = lfai  # expected number of false alarms (integral of lambda_fa)
    model['lambda_fa'] = lfai / volume  # intensity = expected number / state space volume

    simlen = 201  # must be odd
    midpoint = (simlen - 1) // 2
    numfb = midpoint # number of forward backward time steps
    measlog = [None] * simlen # a list to store measurements at each timestep
    xlog = measlog.copy() # a list to store true state information at each timestep

    # Initialize at time midpoint and propagate forward and backward
    x = np.linalg.cholesky(Pmid).T @ np.random.randn(model['F'].shape[0], numtruth) # Genrate initial state samples
    xf = x.copy() # copy for forward simulations
    xb = x.copy() # copy for backwards simulations
    measlog[midpoint] = makemeas(x, model) # record initial measurements
    xlog[midpoint] = x # record initial true states
    
    for t in range(1, numfb+1):
        # Run forward and backward simulation process
        xf = model['F'] @ xf + model['Qc'].T @ np.random.randn(model['F'].shape[0], x.shape[1]) # forward in time using state dynamics and noise
        xb = np.linalg.solve(model['F'], xb + model['Qc'].T @ np.random.randn(model['F'].shape[0], x.shape[1])) # backwards in time using inverse of state dynamics and noise
        measlog[midpoint - t] = makemeas(xb[:, midpoint - t > birthtime], model)
        measlog[midpoint + t] = makemeas(xf, model)
        xlog[midpoint - t] = xb[:, midpoint - t > birthtime]
        xlog[midpoint + t] = xf  # note that all targets exist after midpoint

    return model, measlog, xlog

def makemeas(x, model):
    # Generate target measurements (for every target)
    z = model['H'] @ x + model['Rc'].T @ np.random.randn(model['H'].shape[0], x.shape[1])
    # Simulate missed detection process
    z = z[:, np.random.rand(z.shape[1]) < model['Pd']]
    # Generate false alarms (spatially uniform on [-100, 100]^2)
    z = np.concatenate((z, 200 * np.random.rand(model['H'].shape[0], np.random.poisson(model['lfai'])) - 100), axis=1)
    # Shuffle order
    z = z[:, np.random.permutation(z.shape[1])]
    return z
