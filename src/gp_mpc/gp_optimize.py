# -*- coding: utf-8 -*-
"""
Optimize hyperparameters for Gaussian Process Model
Copyright (c) 2018, Helge-André Langåker
Copyright (c) 2022, Kevin Haninger
"""
from __future__ import (absolute_import, division, print_function)

import time
import numpy as np
import casadi as ca
from scipy.spatial import distance
from .gp_functions import build_mean_func, build_matrices, get_mean_func_hypers
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# Optimization of hyperperameters as a constrained minimization problem
# -----------------------------------------------------------------------------
def calc_NLL(hyper, X, Y, mean_func='zero', prior=None):
    """ Objective function

    Calculate the negative log likelihood function using Casadi SX symbols.

    # Arguments:
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn], where Nx is the
            number of inputs to the GP.

        X: Training data matrix with inputs of size (N x Nx).
        Y: Training data matrix with outpyts of size (N x Ny),
            with Ny number of outputs.

    # Returns:
        NLL: The negative log likelihood function (scalar)
    """
    
    m = build_mean_func(X.shape[0], X.shape[1], Y.shape[1], hyper = hyper, mean_func=mean_func)
    alphas, Ls, invKs = build_matrices(X, Y, hyper, mean_func = mean_func)
    NLL  = 0
    mean = m(X, get_mean_func_hypers(hyper, mean_func))
    for i, (alpha, L, invK) in enumerate(zip(alphas, Ls, invKs)):
        NLL += 0.5 * (Y[:,i]-mean[:,i]).T@alpha
        NLL += 1 * ca.sum1(ca.SX.log(ca.diag(L)))
        #NLL += (Y[:,i]-mean[:,i]).T@invK@(Y[:,i]-mean[:,i])

    # Calculate hyperpriors
    theta = ca.SX.sym('theta')
    mu = ca.SX.sym('mu')
    s2 = ca.SX.sym('s2')
    prior_gauss = ca.Function('hyp_prior', [theta, mu, s2],
                              [-(theta - mu)**2/(2*s2) - 0.5*ca.log(2*ca.pi*s2)])
    log_prior = 0
    if prior is not None:
        for i in range(Nx):
            log_prior += prior_gauss(ell[i], prior['ell_mean'], prior['ell_std']**2)
        log_prior += prior_gauss(sf2, prior['sf_mean'], prior['sf_std']**2)
        log_prior += prior_gauss(sn2, prior['sn_mean'], prior['sn_std']**2)
    return NLL + log_prior

def train_gp(X, Y, hyper_init, mean_func='zero',
             multistart=1, optimizer_opts=None):
    """ Train hyperparameters using CasADi/IPOPT

    Maximum likelihood estimation is used to optimize the hyperparameters of
    the Gaussian Process. The optimalization use CasADi to find the gradients
    and use the interior point method IPOPT to find the solution.

    A uniform prior of the hyperparameters are assumed and implemented as
    limits in the optimization problem.

    NOTE: This function use the symbolic framework from CasADi to optimize the
            hyperparameters, where the gradients are found using algorithmic
            differentiation. This gives the exact gradients, but require a lot
            more memory than the nummeric version 'train_gp_numpy' and have a
            quite horrible scaling problem. The memory usage from the symbolic
            gradients tend to explode with the number of observations.

    # Arguments:
        X: Training data matrix with inputs of size (N x Nx),
            where Nx is the number of inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny),
            with Ny number of outputs.
        meanFunc: String with the name of the wanted mean function.
            Possible options:
                'zero':       m = 0
                'const':      m = a
                'linear':     m(x) = aT*x + b
                'polynomial': m(x) = xT*diag(a)*x + bT*x + c

    # Return:
        opt: Dictionary with the optimal hyperparameters
                [ell_1 .. ell_Nx sf sn].
    """

    N, Nx = X.shape
    Ny = Y.shape[1]

    param_list = hyper_init.keys()
    hyper_delta_sym = {hy: ca.SX.sym(hy+'_sym', *hyper_init[hy].shape) for hy in param_list}

    hyper_total = {hy: hyper_init[hy]+hyper_delta_sym[hy] for hy in param_list}

    # Create solver
    NLL = calc_NLL(hyper_total, X, Y, mean_func=mean_func, prior=None)

    hyper_delta_sym_vec = []
    g = []
    ubx = []
    lbx = []

    for param in param_list:
        param_size = hyper_total[param].shape[0]*hyper_total[param].shape[1]
        hyper_delta_sym_vec += [hyper_delta_sym[param].reshape((param_size,1))]
        g += [hyper_total[param].reshape((param_size,1))]
        if param == 'length_scale':
            ubx += [1.0*np.ones(param_size)]
            lbx += [-np.inf*np.ones(param_size)]
            NLL += 3*ca.sumsqr(hyper_delta_sym_vec[-1])
        elif param in ('noise_var', 'signal_var'):
            ubx += [20*np.ones(param_size)]
            lbx += [-np.inf*np.ones(param_size)]
        else:
            ubx += [np.inf*np.ones(param_size)]
            lbx += [-np.inf*np.ones(param_size)]
    hyper_delta_sym_vec = ca.vertcat(*hyper_delta_sym_vec)
    ubx = ca.vertcat(*ubx)
    lbx = ca.vertcat(*lbx)
    num_hyp = hyper_delta_sym_vec.shape[0]

    g = ca.vertcat(*g)
    lbg = 0.01*np.ones(num_hyp)
    ubg = 30*np.ones(num_hyp)

    #

    nlp = {'x': hyper_delta_sym_vec, 'f': NLL, 'g': g}

    # NLP solver options
    opts = {}
    opts['expand']              = False
    opts['print_time']          = False
    opts['verbose']             = False
    opts['ipopt.print_level']   = 5
    opts['ipopt.constr_viol_tol'] = 1e-14
    opts['ipopt.tol']          = 1e-16
    opts['ipopt.mu_strategy']  = 'adaptive'
    opts['ipopt.nlp_scaling_method'] = 'gradient-based'
    if optimizer_opts is not None:
        opts.update(optimizer_opts)

    opts['ipopt.warm_start_init_point'] = 'yes'
    warm_start = True

    solver = ca.nlpsol('Solver', 'ipopt', nlp, opts)

    print('\n________________________________________')
    print('# Optimizing hyperparameters (N=%d)' % N )
    print('----------------------------------------')

    solve_time = -time.time()
    res = solver(x0=np.zeros(num_hyp), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    status = solver.stats()['return_status']
    obj = res['f']
    solve_time += time.time()

    hyper_opt = {}
    already_read = 0
    for param in param_list:
        param_size = hyper_total[param].shape[0]*hyper_total[param].shape[1]
        hyper_opt[param] = res['x'][already_read:already_read+param_size].reshape(hyper_total[param].shape)
        already_read += param_size

    print("* State %s - %f sec" % (status, solve_time))
    print("Final objective {}".format(obj))
    print("Delta params: {}".format(hyper_opt))

    for param in param_list:
        hyper_opt[param] += hyper_init[param]

    return hyper_opt




# -----------------------------------------------------------------------------
# Optimization of hyperperameters using scipy
# -----------------------------------------------------------------------------

def calc_cov_matrix(X, ell, sf2):
    """ Calculate covariance matrix K

        Squared Exponential ARD covariance kernel

    # Arguments:
        X: Training data matrix with inputs of size (N x Nx).
        ell: Vector with length scales of size Nx.
        sf2: Signal variance (scalar)
    """
    dist = 0
    n, D = X.shape
    for i in range(D):
        x = X[:, i].reshape(n, 1)
        dist = (np.sum(x**2, 1).reshape(-1, 1) + np.sum(x**2, 1) -
                2 * np.dot(x, x.T)) / ell[i]**2 + dist
    return sf2 * np.exp(-.5 * dist)


def calc_NLL_numpy(hyper, X, Y):
    """ Objective function

    Calculate the negative log likelihood function.

    # Arguments:
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn], where Nx is the
                number of inputs to the GP.
        X: Training data matrix with inputs of size (N x Nx).
        Y: Training data matrix with outpyts of size (N x Ny), with Ny number of outputs.

    # Returns:
        NLL: The negative log likelihood function (scalar)
    """

    n, D = X.shape
    ell = hyper[:D]
    sf2 = hyper[D]**2
    lik = hyper[D + 1]**2
    #m   = hyper[D + 2]
    K = calc_cov_matrix(X, ell, sf2)
    K = K + lik * np.eye(n)
    K = (K + K.T) * 0.5   # Make sure matrix is symmentric
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        print("K is not positive definit, adding jitter!")
        K = K + np.eye(n) * 1e-8
        L = np.linalg.cholesky(K)

    logK = 2 * np.sum(np.log(np.abs(np.diag(L))))
    invLy = np.linalg.solve(L, Y)
    alpha = np.linalg.solve(L.T, invLy)
    NLL = 0.5 * np.dot(Y.T, alpha) + 0.5 * logK
    return NLL


def train_gp_numpy(X, Y, meanFunc='zero', hyper_init=None, lam_x0=None, log=False,
             multistart=1, optimizer_opts=None):
    """ Train hyperparameters using scipy / SLSQP

    Maximum likelihood estimation is used to optimize the hyperparameters of
    the Gaussian Process. The optimization use finite differences to estimate
    the gradients and Sequential Least SQuares Programming (SLSQP) to find
    the optimal solution.

    A uniform prior of the hyperparameters are assumed and implemented as
    limits in the optimization problem.

    NOTE: Unlike the casadi version 'train_gp', this function use finite
            differences to estimate the gradients. To get a better result
            and reduce the computation time the explicit gradients should
            be implemented. The gradient equations are given by
            (Rassmussen, 2006).

    NOTE: This version only support a zero-mean function. To enable the use of
            other mean functions, this has to be included in the calculations
            in the 'calc_NLL_numpy' function.

    # Arguments:
        X: Training data matrix with inputs of size (N x Nx),
            where Nx is the number of inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny),
            with Ny number of outputs.
        meanFunc: String with the name of the wanted mean function.
            Possible options:
                'zero':       m = 0
                'const':      m = a
                'linear':     m(x) = aT*x + b
                'polynomial': m(x) = xT*diag(a)*x + bT*x + c

    # Return:
        opt: Dictionary with the optimal hyperparameters [ell_1 .. ell_Nx sf sn].
    """
#    if log:
#        X = np.log(X)
#        Y = np.log(Y)

    N, Nx = X.shape
    Ny = Y.shape[1]

    # Counting mean function parameters
    if meanFunc == 'zero':
        h_m = 0
    elif meanFunc == 'const':
        h_m = 1
    elif meanFunc == 'linear':
        h_m = Nx + 1
    elif meanFunc == 'polynomial':
        h_m = 2 * Nx + 1
    else:
        raise NameError('No mean function called: ' + meanFunc)

    h_ell   = Nx    # Number of length scales parameters
    h_sf    = 1     # Standard deviation function
    h_sn    = 1     # Standard deviation noise
    num_hyp = h_ell + h_sf + h_sn + h_m

    options = {'disp': True, 'maxiter': 20000}
    if optimizer_opts is not None:
        options.update(optimizer_opts)


    hyp_opt = np.zeros((Ny, num_hyp))
    invK = np.zeros((Ny, N, N))
    alpha = np.zeros((Ny, N))
    chol = np.zeros((Ny, N, N))

    print('\n________________________________________')
    print('# Optimizing hyperparameters (N=%d)' % N )
    print('----------------------------------------')
    for output in range(Ny):
        meanF     = np.mean(Y[:, output])
        lb        = -np.inf * np.ones(num_hyp)
        ub        = np.inf * np.ones(num_hyp)

        lb[:Nx]    = 0.0
        ub[:Nx]    = 1e4
        lb[Nx]     = 1e-8
        ub[Nx]     = 1e2
        lb[Nx + 1] = 10**-10
        ub[Nx + 1] = 10**0

        bounds = np.hstack((lb.reshape(num_hyp, 1), ub.reshape(num_hyp, 1)))

        if hyper_init is None:
            hyp_init = np.zeros((num_hyp))
            hyp_init[:Nx] = np.std(X, 0)
            hyp_init[Nx] = np.std(Y[:, output])
            hyp_init[Nx + 1] = 1e-2
        else:
            hyp_init = hyper_init[output, :]

        if meanFunc is 'const':
            lb[-1] = -1e2
            ub[-1] = 1e2
        elif meanFunc is not 'zero':
            lb[-1] = meanF / 10 -1e-8
            ub[-1] = meanF * 10 + 1e-8
            lb[-h_m:-1] = -1e-2
            ub[-h_m:-1] = 1e-2
        obj = np.zeros((multistart, 1))
        hyp_opt_loc = np.zeros((multistart, num_hyp))
        for i in range(multistart):
            solve_time = -time.time()
            res = minimize(calc_NLL_numpy, hyp_init, args=(X, Y[:, output]),
                           method='SLSQP', options=options, bounds=bounds, tol=1e-8)
            obj[i] = res.fun
            hyp_opt_loc[i, :] = res.x
        solve_time += time.time()
        print("* State %d:  %f s" % (output, solve_time))

        # With multistart, get solution with lowest decision function value
        hyp_opt[output, :]   = hyp_opt_loc[np.argmin(obj)]
        ell = hyp_opt[output, :Nx]
        sf2 = hyp_opt[output, Nx]**2
        sn2 = hyp_opt[output, Nx + 1]**2

        # Calculate the inverse covariance matrix
        K = calc_cov_matrix(X, ell, sf2)
        K = K + sn2 * np.eye(N)
        K = (K + K.T) * 0.5   # Make sure matrix is symmentric
        try:
            L = np.linalg.cholesky(K, dtype=np.single)
        except np.linalg.LinAlgError:
            print("K matrix is not positive definit, adding jitter!")
            K = K + np.eye(N) * 1e-8
            L = np.linalg.cholesky(K).astype(np.single)
        invL = np.linalg.solve(L, np.eye(N)).astype(np.single)
        invK[output, :, :] = np.linalg.solve(L.T, invL).astype(np.single)
        chol[output] = L
        m = get_mean_function(ca.MX(hyp_opt[output, :]), X.T, func=meanFunc)
        mean = np.array(m(X.T)).reshape((N,))
        alpha[output] = np.linalg.solve(L.T, np.linalg.solve(L, Y[:, output] - mean)).astype(np.single)
    print('----------------------------------------')

    opt = {}
    opt['hyper'] = hyp_opt
    opt['lam_x'] = 0 # Warm start not implemented
    opt['invK'] = invK
    opt['alpha'] = alpha
    opt['chol'] = chol
    return opt



# -----------------------------------------------------------------------------
# Validation of model
# -----------------------------------------------------------------------------

def validate(X_test, Y_test, X, Y, invK, hyper, meanFunc, alpha=None):
    """ Validate GP model with new test data
    """
    N, Ny = Y_test.shape
    Nx = np.size(X, 1)
    z_s = ca.MX.sym('z', Nx)

    gp_func = ca.Function('gp', [z_s],
                                gp(invK, ca.MX(X), ca.MX(Y), ca.MX(hyper),
                                   z_s, meanFunc=meanFunc, alpha=alpha))
    loss = 0
    NLP = 0

    for i in range(N):
        mean, var = gp_func(X_test[i, :])
        loss += (Y_test[i, :] - mean)**2
        NLP += 0.5*np.log(2*np.pi * (var)) + ((Y_test[i, :] - mean)**2)/(2*var)
        print(NLP)
        print(var)
    loss = loss / N
    SMSE = loss/ np.std(Y_test, 0)
    MNLP = NLP / N


    print('\n________________________________________')
    print('# Validation of GP model ')
    print('----------------------------------------')
    print('* Num training samples: ' + str(np.size(Y, 0)))
    print('* Num test samples: ' + str(N))
    print('----------------------------------------')
    print('* Mean squared error: ')
    for i in range(Ny):
        print('\t- State %d: %f' % (i + 1, loss[i]))
    print('----------------------------------------')
    print('* Standardized mean squared error:')
    for i in range(Ny):
        print('\t* State %d: %f' % (i + 1, SMSE[i]))
    print('----------------------------------------\n')
    print('* Mean Negative log Probability:')
    for i in range(Ny):
        print('\t* State %d: %f' % (i + 1, MNLP[i]))
    print('----------------------------------------\n')
    return SMSE, MNLP


"""-----------------------------------------------------------------------------
# Preprocesing of training data
-----------------------------------------------------------------------------"""


def normalize(X, lb, ub):
    """ Normalize data between 0 and 1
    # Arguments:
        X: Input data (scalar/vector/matrix)
        lb: Lower boundry (scalar/vector)
        ub: Upper boundry (scalar/vector)
    # Return:
        X normalized (scalar/vector/matrix)
    """

    return (X - lb) / (ub - lb)


def normalize_inverse(X_scaled, lb, ub):
    # Scale input and output variables
    # Normalize input data to [0 1]
    return X_scaled * (ub - lb) + lb


def standardize(X_original, meanX, stdX):
    # Scale input and output variables
    return (X_original - meanX) / stdX


def standardize_inverse(X_scaled, meanX, stdX):
    # Scale input and output variables
    return X_scaled * stdX + meanX
