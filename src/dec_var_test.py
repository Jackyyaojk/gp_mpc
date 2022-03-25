import casadi as ca
import numpy as np
from decision_vars import decision_var, decision_var_set

dec_vars = decision_var_set(ca.SX.sym)
dec_vars['test1'] = decision_var(np.array([[1.0, 1.0, 1.0],[2.0, 2.0, 2.0]]), ub = np.array([1, 2, 1]))
dec_vars['test2'] = decision_var(np.array([[3.0, 4.0, 5.0],[6.0, 7.0, 8.0]]))
print(dec_vars.get_decision_vector())
print(dec_vars['test1'])
dec_vars.set_results(np.ones((12,1)))

print(dec_vars['test1'])
