"""
Copyright (c) 2022, Kevin Haninger
Helper classes for decision variables in an optimization problem
"""

import casadi as ca
import numpy as np
from sys import version_info

class decision_var:
    """
    Individual optimization variable, initialized from an initial value x0
    """
    def __init__(self, x0, lb = -np.inf, ub = np.inf):
        self.x0 = x0
        self.shape = x0.shape
        self.size = x0.size
        self.lb = np.full(self.shape, lb)
        self.ub = np.full(self.shape, ub)

        assert np.all(x0 >= self.lb), "x0 below given lower bound"
        assert np.all(x0 <= self.ub), "x0 above given upper bound"

    def __len__(self):
        return self.size

class decision_var_set:
    """
    Helper class for sets of decision variables.
    Key functionalities:
      - Add with dec_var_set[key] = decision_var(x0)
      - Access the symbolic value dec_var_set[key]
      - Set optimized results with dec_var_set.set_results(x_opt)
      - Access numerical optimized values with dec_var_set[key]
    """
    def __init__(self, var_type = ca.SX.sym, x0 = {}, lb = {}, ub = {}):
        """
        Arguments:
          - var_type: the type of symbolic optimization variable which should be constructed
          - x0: optional dict of initial values, will construct decision_vars for all keys
          - ub: optional dict of upper bounds, if no key for a key in x0, will default to  np.inf
          - lb: optional dict of lower bounds, if no key for a key in x0, will default to -np.inf
        """
        assert version_info >= (3, 6), "Python 3.6 required to guarantee dicts are ordered"
        self.__ty = var_type
        self.__vars = {}
        self.results = dict()
        for key in x0.keys():
            self[key] = decision_var(x0[key],
                                     lb = lb.get(key, -np.inf),
                                     ub = ub.get(key, np.inf))

    def __setitem__(self, key, value):
        """
        # Arguments:
            key: name of variable
            value: dec_var
        """
        value.x = self.__ty(key, *value.shape)
        self.__vars[key] = value
        self.__keys = list(self.__vars.keys())

    def __getitem__(self, key):
        """
        If no results are set, returns the symbolic variable at key
        If results are set, returns the numeric value at key
        """
        if not self.results:
            return self.__vars[key].x
        else:
            return self.results[key]

    def __len__(self):
        return sum(len(val) for val in self.__vars.values())

    def __str__(self):
        s = "** Decision variables **\n"
        for key in self.__keys:
            s += "{}:\n: {}\n".format(key, self[key])
        return s

    def vectorize(self, attr):
        return ca.vertcat(*[getattr(el, attr).reshape((el.size,1)) for el in self.__vars.values()])

    def get_dec_vectors(self):
        """
        Returns a tuple that you need to get that optimization problem going
        """
        x  = self.vectorize('x')
        lb = self.vectorize('lb')
        ub = self.vectorize('ub')
        return x, lb, ub

    def get_x0(self):
        return self.vectorize('x0')

    def get_deviation(self, key):
        """
        Returns difference between initial value and symbolic (or numeric) value
        """
        return self[key]-self.__vars[key].x0

    def set_results(self, x_opt):
        """
        x_opt is the numerical optimization results, fills the dict x with reshaping as needed
        """
        x_opt = x_opt.full()
        assert len(x_opt) is len(self), "Length of optimization doesn't match initial x0"
        read_pos = 0
        for key in self.__keys:
            v_size  = self.__vars[key].size
            v_shape = self.__vars[key].shape
            self.results[key] = x_opt[read_pos:read_pos+v_size].reshape(v_shape)
            read_pos += v_size
        return self.results


