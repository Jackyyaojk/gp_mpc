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
from .gp_functions import build_mean_func, build_matrices, get_mean_func_hypers
from decision_vars import decision_var_set

def calc_NLL(hyper, X, Y, mean_func='zero'):
    """ Calculate the negative log likelihood function using Casadi SX symbols.

    # Arguments:
        hyper: dictionary of hyperparameters

        X: Training data matrix with inputs of size (N x Nx).
        Y: Training data matrix with outpyts of size (N x Ny),
            with Ny number of outputs.

        mean_func: string which specifiesx mean_function

    # Returns:
        NLL: The negative log likelihood function (scalar)
    """

    m = build_mean_func(X.shape[0], X.shape[1], Y.shape[1], hyper = hyper, mean_func = mean_func)
    alphas, Ls, invKs = build_matrices(X, Y, hyper, mean_func = mean_func)
    NLL  = 0
    mean = m(X, get_mean_func_hypers(hyper, mean_func))
    for i, (alpha, L, invK) in enumerate(zip(alphas, Ls, invKs)): 
        NLL += 0.5 * (Y[:,i]-mean[:,i]).T@alpha
        NLL += 1 * ca.sum1(ca.SX.log(ca.diag(L)))
        #NLL += (Y[:,i]-mean[:,i]).T@invK@(Y[:,i]-mean[:,i])

    return NLL

def train_gp(X, Y, hyper_init, mean_func='zero', opts={}):
    """ Train hyperparameters using IPOPT

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

    # Returns:
        opt: Dictionary with the optimized hyperparameters
    """

    N, Nx = X.shape
    Ny = Y.shape[1]

    hyper_lb = {p:0.01 for p in ('length_scale', 'noise_var', 'signal_var')}
    hyper_ub = {'length_scale':1.0, 'noise_var':20.0, 'signal_var':20.0}
    dec_vars = decision_var_set(x0 = hyper_init, lb = hyper_lb, ub = hyper_ub)

    # Create solver
    loss = calc_NLL(dec_vars, X, Y, mean_func = mean_func)

    # Penalize the deviation in length scale
    loss += 3*ca.sumsqr(dec_vars.get_deviation('length_scale'))
    loss += ca.sumsqr(dec_vars.get_deviation('noise_var'))
    loss += ca.sumsqr(dec_vars.get_deviation('signal_var'))

    x, lbx, ubx = dec_vars.get_dec_vectors()

    nlp = {'x': x, 'f': loss}

    # NLP solver options
    opts['expand']              = False
    opts['print_time']          = False
    opts['verbose']             = False
    opts['ipopt.print_level']   = 3
    opts['ipopt.constr_viol_tol'] = 1e-14
    opts['ipopt.tol']          = 1e-16
    opts['ipopt.mu_strategy']  = 'adaptive'
    opts['ipopt.nlp_scaling_method'] = 'gradient-based'
    opts['ipopt.warm_start_init_point'] = 'yes'

    solver = ca.nlpsol('Solver', 'ipopt', nlp, opts)

    print('\n________________________________________')
    print('# Optimizing hyperparameters (N=%d)' % N )
    print('----------------------------------------')

    solve_time = -time.time()
    res = solver(x0=dec_vars.get_x0(), lbx=lbx, ubx=ubx)
    status = solver.stats()['return_status']
    obj = res['f']
    hyper_opt = dec_vars.set_results(res['x'])
    solve_time += time.time()

    print("* State %s - %f sec" % (status, solve_time))
    print("Final objective {}".format(obj))
    print(dec_vars)

    return hyper_opt

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
