"""
Helper class for decision variables in an optimization problem


NOTE: we assume python --version >= 3.6, so that the dict is ordered
"""

import casadi as ca
import numpy as np

class decision_var:
    def __init__(self, x0, lb = -np.inf, ub = np.inf):
        self.x0 = x0
        self.shape = x0.shape
        self.size = x0.size
        self.lb = np.full(self.shape, lb)
        self.ub = np.full(self.shape, ub)

class decision_var_set:
    """
    Helper class for sets of decision variables.
    Key functionalities:
      - Add with dec_var_set[key] = decision_var(x0)
      - Access the symbolic value dec_var_set[key]
      - Set optimized results with dec_var_set.set_results(x_opt)
      - Access numerical optimized values with dec_var_set[key]
    """
    def __init__(self, var_type):
        self.__ty = var_type
        self.__vars = {}
        self.results = dict()

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
        if not self.results:
            return self.__vars[key].x
        else:
            return self.results[key]

    def vectorize(self, attr):
        return ca.vertcat(*[getattr(el, attr).reshape((el.size,1)) for el in self.__vars.values()])

    def get_decision_vector(self):
        x  = self.vectorize('x')
        x0 = self.vectorize('x0')
        lb = self.vectorize('lb')
        ub = self.vectorize('ub')
        return x, x0, lb, ub

    def set_results(self, x_opt):
        read_pos = 0
        for key in self.__keys:
            v_size  = self.__vars[key].size
            print(v_size)
            v_shape = self.__vars[key].shape
            self.results[key] = x_opt[read_pos:read_pos+v_size].reshape(v_shape)
            read_pos += v_size

    

