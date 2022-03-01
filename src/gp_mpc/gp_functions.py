# -*- coding: utf-8 -*-
"""
Gaussian Process functions
Copyright (c) 2018, Helge-André Langåker, Eric Bradford
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (
         bytes, dict, int, list, object, range, str,
         ascii, chr, hex, input, next, oct, open,
         pow, round, super,
         filter, map, zip)

import numpy as np
import casadi as ca

def covSEard_fn(ell, sf2):
    D = ell.shape[0]
    x = ca.SX.sym('x', D, 1)
    z = ca.SX.sym('z', D, 1)
    func = ca.Function('cov', [x, z], [sf2 * ca.SX.exp(-0.5 * ca.sum1((x-z)**2 / ell**2))])
    return func

def build_mean_func(X, mean_hyper = [], mean_func='zero'):
    """ Get mean function
        Copyright (c) 2018, Helge-André Langåker

    # Arguments:
        hyper: Matrix with hyperperameters.
        X: Input vector or matrix.
        func: Option for mean function:
                'zero':       m = 0
                'const':      m = a
                'linear':     m(x) = aT*x + b
                'polynomial': m(x) = aT*x^2 + bT*x + c
    # Returns:
         CasADi mean function [m(X, hyper)]
    """

    Nx, N = X.shape
    X_s = ca.SX.sym('x', Nx, N)
    Z_s = ca.SX.sym('x', Nx, N)
    m = ca.SX(N, 1)
    hyp_s = ca.SX.sym('hyper', *np.array(mean_hyper).shape)
    if mean_func == 'zero':
        meanF = ca.Function('zero_mean', [X_s, hyp_s], [m])
    elif mean_func == 'const':
        a =  hyp_s[-1]
        for i in range(N):
            m[i] = a
        meanF = ca.Function('const_mean', [X_s, hyp_s], [m])
    elif mean_func == 'linear':
        a = hyp_s[-Nx-1:-1].reshape((1, Nx))
        b = hyp_s[-1]
        for i in range(N):
            m[i] = ca.mtimes(a, X_s[:,i]) + b
        meanF = ca.Function('linear_mean', [X_s, hyp_s], [m])
    elif mean_func == 'polynomial':
        a = hyp_s[-2*Nx-1:-Nx-1].reshape((1,Nx))
        b = hyp_s[-Nx-1:-1].reshape((1,Nx))
        c = hyp_s[-1]
        for i in range(N):
            m[i] = ca.mtimes(a, X_s[:, i]**2) + ca.mtimes(b, X_s[:,i]) + c
        meanF = ca.Function('poly_mean', [X_s, hyp_s], [m])
    else:
        raise NameError('No mean function called: ' + func)

    return ca.Function('mean', [Z_s], [meanF(Z_s, mean_hyper)])

def build_gp(invK, X, hyper, alpha, chol, fast_gp_axis, mean_func='zero', jit_opts = {}):
    """ Build Gaussian Process function optimized for comp. graph and speed
        - hyper are consts, not symbolics
        - alpha is a const, not symbolic
        - 0 mean
        Copyright (c) 2021, Kevin Haninger

    # Arguments
        invK: Array with the inverse covariance matrices of size (Ny x N x N),
            with Ny number of outputs from the GP and N number of training points.
        X: Training data matrix with inputs of size (N x Nx), with Nx number of
            inputs to the GP.
        alpha: Training data matrix with invK time outputs of size (Ny x N).
        hyper: Array with hyperparame|ters [ell_1 .. ell_Nx sf sn]
               These are now constants, not symbolics
    # Returns
        mean:     GP mean casadi function [mean(z)]
        var:      GP variance casadi function [var(z)]
        covar:    GP covariance casadi function [covar(z) = diag(var(z))]
        mean_jac: Casadi jacobian of the GP mean function [jac(z)]
    """
    Ny = len(invK)
    N = X.shape[0]
    Nx = X.shape[1]

    mean  = ca.SX.zeros(Ny, 1)
    var   = ca.SX.zeros(Ny, 1)
    z_s   = ca.SX.sym('z', Nx)    # test data state

    for output in range(Ny):
        ell      = hyper['length_scale'][output]
        sf2      = hyper['signal_var'][output]
        sn2      = hyper['noise_var'][output]
        alpha_a  = alpha[output]
        covSE =  covSEard_fn(ell, sf2)

        ks_func  = covSE
        ks       = ks_func(X.T, z_s)
        v        = ca.solve(chol[output], ks.T)
        mean[output] = ks@alpha_a
        var[output]  = sn2 + sf2 - ca.mtimes(v.T, v)
    mean_func  = ca.Function('mean', [z_s], [mean], jit_opts)
    var_func = ca.Function('var', [z_s], [var], jit_opts)
    var_red_func = ca.Function('var_red', [z_s], [var[fast_gp_axis]], jit_opts)
    mean_jac = ca.Function('mean_jac_z', [z_s], [ca.jacobian(mean_func(z_s), z_s)], jit_opts)

    return mean_func.expand(), var_func.expand(), mean_jac.expand(), var_red_func.expand()

def build_TA_cov(mean, covar, jac, Nx, Ny):
    """ Build 1st order Taylor approximation of covariance function
        Copyright (c) 2018, Helge-André Langåker

    # Arguments:
        mean: GP mean casadi function [mean(z)]
        covar: GP covariance casadi function [covar(z)]
        jac: Casadi jacobian of the GP mean function [jac(z)]
        Nx: Number of inputs to the GP
        Ny: Number of ouputs from the GP

    # Return:
        cov: Casadi function with the approximated covariance
             function [cov(z, covar_x)].
    """
    cov_z  = ca.SX.sym('cov_z', Nx, Nx)
    z_s    = ca.SX.sym('z', Nx)
    jac_z = jac(z_s)
    cov    = ca.Function('cov', [z_s, cov_z],
                      [covar(z_s) + ca.mtimes(ca.mtimes(jac_z, cov_z), jac_z.T)])

    return cov

def build_const_matrices(X, Y, hyper, mean_func):
        N, D = X.shape
        Ny = Y.shape[1]

        invK  = np.zeros((Ny, N , N ))
        K     = np.zeros((N, N ))
        alpha = np.zeros((Ny, N ))
        chol  = np.zeros((Ny, N , N ))

        for output in range(Ny):
            ell = hyper['length_scale'][output]
            sf2 = hyper['signal_var'][output]
            sn2 = hyper['noise_var'][output]

            K_fn =  covSEard_fn(ell, sf2)
            for i in range(X.shape[0]):
                K[i, :] = K_fn(X.T, X[i,:])
            K    = K + sn2 * np.eye(N)
            K    = (K + K.T) * 0.5   # Make sure matrix is symmentric

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                print("K matrix is not positive definite, adding jitter!")
                K = K + np.eye(N) * 1e-8
                L = np.linalg.cholesky(K)

            invL = np.linalg.solve(L, np.eye(N))
            invK[output, :, :] = np.linalg.solve(L.T, invL)
            chol[output] = L
            m = build_mean_func(X.T, mean_func = mean_func)
            mean = np.array(m(X.T)).reshape((N,))
            alpha[output] = np.linalg.solve(L.T, np.linalg.solve(L, Y[:, output] - mean))
        return alpha, chol, invK

def normalize(Y, mean, std):
    return (Y - mean) / std

def inv_normalize(Y, mean, std):
    return (Y * std) + mean
