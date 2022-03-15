# -*- coding: utf-8 -*-
"""
Gaussian Process Model
Copyright (c) 2018, Helge-André Langåker
Revised 2021, Christian Hegeler & Kevin Haninger
"""
from __future__ import (absolute_import, division, print_function)

import numpy as np
import casadi as ca
from .gp_functions import build_gp, build_TA_cov, build_mean_func, build_matrices
from .gp_functions import normalize, inv_normalize
from .optimize import train_gp


class GP:
    def __init__(self, X, Y, hyper, opt_hyper = False, mean_func="zero",
                 gp_method="ME", normalize=False, fast_axis = 0 ):
        """ Initialize a GP model.
        # Arguments:
            X: State data (N x Nx)
            Y: Observation data (N x Ny)
            hyper: dictionary with hyperparams
               length_scale: \ell (not squared)
               signal_var:   \sigma_f^2, i.e. directly used in kernel
               noise_var:    \sigma_n^2, i.e. directly used in kernel
        """
        self.dtype = np.single

        self.__hyper = hyper
        self.__gp_method = gp_method
        self.__mean_func = mean_func
        self.__normalize = normalize
        self.__fast_axis = fast_axis

        self.init_data(X, Y)

        self.train_model(opt_hyper = opt_hyper)

        self.build_model()

    def init_data(self, X, Y):
        """
        Copy in data and normalize if requested
        """
        self.__X = X.astype(self.dtype)
        self.__Y = Y.astype(self.dtype)
        self.__Ny = Y.shape[1]
        self.__Nx = X.shape[1]
        self.__N = X.shape[0]

        if self.__normalize:
            self.__meanX = np.mean(self.__X, axis = 0)
            self.__stdX  = np.std(self.__X, axis = 0)
            self.__X = normalize(X, self.__meanX, self.__stdX)
            print("Normalizing x \n  mean: {} \n   std: {}".format(self.__meanX, self.__stdX))

            self.__meanY = np.mean(self.__Y, axis = 0)
            self.__stdY  = np.std(self.__Y, axis = 0)
            self.__Y = normalize(Y, self.__meanY, self.__stdY)
            print("Normalizing y \n  mean: {} \n   std: {}".format(self.__meanY, self.__stdY))

    def train_model(self, opt_hyper):
        if opt_hyper: self.__hyper = train_gp(self.__X, self.__Y, self.__hyper, mean_func = self.__mean_func)

    def build_model(self):
        """
        Calculate cholesky covariance matrix, alpha, inverse covariance, and build
        the GP mean/var functions
        """
        self.__alpha, self.__chol, self.__invK = \
            build_matrices(self.__X, self.__Y, self.__hyper, self.__mean_func)

        self.__mean, self.__var,  self.__mean_jac, self.__var_red = \
                            build_gp(self.__invK, self.__X, self.__hyper,
                                     self.__alpha, self.__chol, self.__fast_axis)

        self.set_method(self.__gp_method)

    def set_method(self, gp_method='ME'):
        """ Select which GP covariance approximation to use
        # Arguments:
            gp_method: Method for propagating uncertainty.
                        'ME': Mean Equivalence (normal GP)
                        'TA': 1st order Tayolor Approximation
        """

        x = ca.MX.sym('x', self.__Nx)
        covar_s = ca.MX.sym('covar', self.__Nx, self.__Nx)
        self.__gp_method = gp_method

        if gp_method == 'ME':
            self.__predict = ca.Function('gp_mean', [x, covar_s],
                                         [self.__mean(x), self.__var(x)])
        elif gp_method == 'TA':
            self.__TA_covar = build_TA_cov(self.__mean, self.__var,
                                           self.__mean_jac, self.__Nx, self.__Ny)
            self.__predict = ca.Function('gp_taylor', [x, covar_s],
                                         [self.__mean(x),
                                          self.__TA_covar(x, covar_s)])
        else:
            raise NameError('No GP method called: ' + gp_method)

    def predict(self, x, cov = [], fast = False):
        """ Predict GP output
        # Arguments:
            x: State vector (Nx x 1)
            cov: Covariance matrix of input x
        """
        if fast: return self.predict_fast(x)
        if self.__normalize:
            x_s = normalize(x, self.__meanX, self.__stdX)
        else:
            x_s = x
        mean, cov = self.__predict(x_s, cov)
        if self.__normalize:
            mean = inv_normalize(mean, self.__meanY, self.__stdY)
            cov = cov*self.__stdY**2
        return mean, cov

    def predict_fast(self, x):
        if self.__normalize:
            print("Predict fast *only* for un-normalized models")
            return
        return self.__mean(x), self.__var_red(x)

    def log_lik(self, X = None, Y = None):
        liks = []
        if X is None and Y is None:
            for out in range(self.__Ny):
                lik = 0
                L = ca.SX(self.__chol[out])
                lik += -0.5*self.__Y[:,out]@self.__alpha[out]
                for n in range(self.__N):
                    lik += -ca.log(self.__chol[out][n,n])
                liks.append(lik)
        return liks

    def grad(self, x):
        return self.__mean_jac(x)

    def get_xrange(self):
        if self.__normalize:
            minx = inv_normalize(np.min(self.__X, axis=0), self.__meanX, self.__stdX)
            maxx = inv_normalize(np.max(self.__X, axis=0), self.__meanX, self.__stdX)
        else:
            minx = np.min(self.__X, axis=0)
            maxx = np.max(self.__X, axis=0)
        return [minx, maxx]

    def get_mean_state(self):
        return np.mean(self.__X, axis = 0)

    def get_hyper_parameters(self):
        return self.__hyper

    def set_hyper_parameters(self, hyper):
        self.__hyper = hyper

    def print_hyper_parameters(self):
        """ Print out all hyperparameters """
        print(self.__hyper)

    def mean_jacobian(self, x0):
        """ Jacobian of posterior mean """
        return self.__mean_jac(x0)

    def get_data(self):
        if self.__normalize:
            return inv_normalize(self.__X, self.__meanX, self.__stdX), \
                   inv_normalize(self.__Y, self.__meanY, self.__stdY)
        return self.__X, self.__Y

    def validate(self, X_test, Y_test):
        """ Validate GP model with test data
        """
        Y_test = Y_test.copy()
        X_test = X_test.copy()

        N, Ny = Y_test.shape
        loss = 0
        NLP = 0

        for i in range(N):
            mean = self.__mean(X_test[i, :])
            var = self.__var(X_test[i, :]) + self.noise_variance()
            loss += (Y_test[i, :] - mean)**2
            NLP += 0.5*np.log(2*np.pi * (var)) + ((Y_test[i, :] - mean)**2)/(2*var)

        loss = loss / N
        SMSE = loss/ np.std(Y_test, 0)
        MNLP = NLP / N

        print('\n________________________________________')
        print('# Validation of GP model ')
        print('----------------------------------------')
        print('* Num training samples: ' + str(self.__N))
        print('* Num test samples: ' + str(N))
        print('----------------------------------------')
        print('* Mean squared error: ')
        for i in range(Ny):
            print('\t- State %d: %f' % (i + 1, loss[i]))
        print('----------------------------------------')
        print('* Standardized mean squared error:')
        for i in range(Ny):
            print('\t* State %d: %f' % (i + 1, SMSE[i]))
        print('----------------------------------------')
        print('* Mean Negative log Probability:')
        for i in range(Ny):
            print('\t* State %d: %f' % (i + 1, MNLP[i]))
        print('----------------------------------------\n')

        self.__SMSE = np.max(SMSE)

        return np.array(SMSE).flatten(), np.array(MNLP).flatten()

