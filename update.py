# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:46:32 2023

This code has been translated using chatGPT - 3.5 from the original MATLAB scripts
provided as ancillary files to https://arxiv.org/abs/1203.2995 by Jason L. Williams.

@author: sggjone5
"""

import numpy as np

def update(lambdau, xu, Pu, r, x, P, z, model):
    # Extract parameters from the model
    Pd = model["Pd"]  # Detection probability
    H = model["H"]  # Measurement matrix
    R = model["R"]  # Measurement noise covariance
    lambda_fa = model["lambda_fa"]  # False alarm intensity
    lambdab_threshold = 1e-4  # Threshold for low birth intensity

    # Get the number of existing tracks and measurements
    n = len(r)
    stateDimensions, nu = xu.shape[0], xu.shape[1]
    measDimensions, m = z.shape[0], z.shape[1]

    # Initialize arrays for storing updated values
    wupd = np.zeros((n, m + 1))
    rupd = np.zeros((n, m + 1))
    xupd = np.zeros((m + 1, stateDimensions, n))
    Pupd = np.zeros((n, m + 1, stateDimensions, stateDimensions))

    wnew = np.zeros(m)
    rnew = np.zeros(m)
    xnew = np.zeros((stateDimensions, m))
    Pnew = np.zeros((m, stateDimensions, stateDimensions))

    Sk = np.zeros((nu, measDimensions, measDimensions))
    Kk = np.zeros((nu, stateDimensions, measDimensions))
    Pk = np.zeros((nu, stateDimensions, stateDimensions))
    ck = np.zeros(nu)
    sqrt_det2piSk = np.zeros(nu)
    yk = np.zeros((stateDimensions, nu))

    for i in range(n):
        # Update existing tracks and measurements
        wupd[i, 0] = 1 - r[i] + r[i] * (1 - Pd)
        rupd[i, 0] = r[i] * (1 - Pd) / wupd[i, 0]
        xupd[0, :, i] = x[:, i]
        Pupd[i, 0, :, :] = P[i, :, :]

        # Calculate measurement-related values
        S = np.dot(np.dot(H, P[i, :, :]), H.T) + R
        sqrt_det2piS = np.sqrt(np.linalg.det(2 * np.pi * S))
        K = np.dot(np.dot(P[i, :, :], H.T), np.linalg.inv(S))
        Pplus = P[i, :, :] - np.dot(np.dot(K, H), P[i, :, :])

        for j in range(m):
            v = z[:, j] - np.dot(H, x[:, i])
            wupd[i, j + 1] = r[i] * Pd * np.exp(-0.5 * np.dot(v.T, np.dot(np.linalg.inv(S), v))) / sqrt_det2piS
            rupd[i, j + 1] = 1
            xupd[j + 1, :, i] = x[:, i] + np.dot(K, v)
            Pupd[i, j + 1, :, :] = Pplus

    for k in range(nu):
        # Calculate parameters for PPP components
        Sk[k, :, :] = np.dot(np.dot(H, Pu[k, :, :]), H.T) + R
        sqrt_det2piSk[k] = np.sqrt(np.linalg.det(2 * np.pi * Sk[k, :, :]))
        Kk[k, :, :] = np.dot(np.dot(Pu[k, :, :], H.T), np.linalg.inv(Sk[k, :, :]))
        Pk[k, :, :] = Pu[k, :, :] - np.dot(np.dot(Kk[k, :, :], H), Pu[k, :, :])

    for j in range(m):
        # Update PPP components using measurements
        ck = np.zeros(nu)
        for k in range(nu):
            v = z[:, j] - np.dot(H, xu[:, k])
            ck[k] = lambdau[k] * Pd * np.exp(-0.5 * np.dot(v.T, np.dot(np.linalg.inv(Sk[k, :, :]), v))) / sqrt_det2piSk[k]
            yk[:, k] = xu[:, k] + np.dot(Kk[k, :, :], v)

        C = np.sum(ck)
        wnew[j] = C + lambda_fa
        rnew[j] = C / wnew[j]
        ck = ck / C
        xnew[:, j] = np.dot(yk, ck)

        for k in range(nu):
            v = xnew[:, j] - yk[:, k]
            Pnew[j, :, :] = Pnew[j, :, :] + ck[k] * (Pk[k, :, :] + np.outer(v, v))

    lambdau = (1 - Pd) * lambdau

    # Not shown in the paper -- truncate low weight components
    ss = lambdau > lambdab_threshold
    lambdau = lambdau[ss]
    xu = xu[:, ss]
    Pu = Pu[ss, :, :]

    return lambdau, xu, Pu, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew

