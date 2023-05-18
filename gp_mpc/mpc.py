# Library includes
import numpy as np
import casadi as ca

# Custom includes
from .gp_model import GP
from .helper_fns import yaml_load, constraints, compliance_to_world
from .decision_vars import *


class MPC:
    '''
    This class builds a multiple shooting MPC problem into a solver
    IN:
      N_p: number of shooting points
      mpc_params: the dict for the MPC parameters
      gp_dynamics_dict: the dict of different dynamics objects 
    '''
    def __init__(self, N_p, mpc_params, gp_dynamics_dict, path):
        self.mpc_params = mpc_params
        self.options = yaml_load(path, 'ipopt_params.yaml')
        
        # Set up problem dimensions
        self.__N = mpc_params['mpc_pts']                  # number of MPC steps
        self.__dt = mpc_params['dt']                      # sample time
        self.__N_x = (2+2*mpc_params['state_cov'])*N_p    # number of states of ode
        self.__N_p = N_p
    

        self.__gp_dynamics = gp_dynamics_dict
        self.__modes = gp_dynamics_dict.keys()            # names of modes
        #self.__dyn = {mode:gp_dynamics_dict[mode].build_dynamics().map(self.__N, 'serial') for mode in self.__modes}
        self.__dyn = {mode:gp_dynamics_dict[mode].build_dynamics() for mode in self.__modes}
        
        
    def solve(self, params):
        # params are the numerical values of initial robot pose, mode belief, and impedance parameters

        #Create problem and solver
        if not hasattr(self, "solver"):
            self.build_solver(params)

        # Update parameters for the solver
        self.args['p'] = self.__params.update(params)

        # Solve the NLP
        sol = self.solver(**self.args)

        # Save solution + lagrangian for warm start
        self.args['x0'] = sol['x']
        self.args['lam_x0'] = sol['lam_x']
        self.args['lam_g0'] = sol['lam_g']

        self.__vars.set_results(sol['x'])
        return self.__vars.get_dec_dict()

    # Formulate the NLP for multiple-shooting
    def build_solver(self, params0): 
        N_x = self.__N_x
        N_u = self.__N_p

        # Initialize empty NLP
        J = {mode:0.0 for mode in self.__modes}  # Objective function
        g = []      # constraints functions
        lbg = []    # lower bound on constraints
        ubg = []    # upper-bound on constraints
        vars0 = {} # initial values for decision variables

        # Symbolic varaibles for parameters, these get assigned to numerical values in solve()
        self.__params = ParamSet(params0)

        # Build decision variables
        imp_mass = self.mpc_params['imp_mass']*np.ones(3)
        vars0['imp_stiff'] = self.__params['imp_stiff'] # imp stiff in tcp coord
        vars0['des_pose'] = np.zeros((N_u))             # rest position of imp spring relative to tcp
        for m in self.__modes:
            vars0['x_'+m] = np.zeros((N_x, self.__N-1)) # trajectory relative to tcp
        ub, lb = self.build_dec_var_constraints()
        
        # Turn decision variables into a dec_var object
        self.__vars = DecisionVarSet(x0 = vars0, ub = ub, lb = lb)

        if self.mpc_params['opti_MBK']:
            g += [self.__vars.get_deviation('imp_stiff')]
            lbg += [-self.mpc_params['delta_K_max']]*self.__N_p
            ubg += [self.mpc_params['delta_K_max']]*self.__N_p
            g += [self.__vars['des_pose']]
            lbg += [-self.mpc_params['delta_xd_max']]*self.__N_p
            ubg += [self.mpc_params['delta_xd_max']]*self.__N_p

        for mode in self.__modes:
            dyn_next = self.__dyn[mode](x = ca.horzcat(np.zeros((N_x, 1)), self.__vars['x_'+mode]),
                                        des_pose = self.__vars['des_pose'],
                                        init_pose = self.__params['pose'],
                                        imp_mass = imp_mass,
                                        imp_damp = ca.DM([20, 20, 20]),#2*ca.sqrt(self.__vars['imp_stiff']),
                                        imp_stiff = ca.DM([100, 100, 100]))#self.__vars['imp_stiff'])
            x_next = dyn_next['x_next']
            J[mode] += ca.sum2(dyn_next['st_cost'])

            g += [ca.reshape(x_next[:,:-1]-self.__vars['x_'+mode][:,:], N_x*(self.__N-1), 1)]
            lbg += [ self.mpc_params['constraint_slack']]*N_x*(self.__N-1)
            ubg += [-self.mpc_params['constraint_slack']]*N_x*(self.__N-1)

        # Calculate total objective
        J_total = self.mpc_params['delta_xd_cost']*ca.sumsqr(self.__vars['des_pose'])
        if self.mpc_params['opti_MBK']:
            J_total += self.mpc_params['delta_K_cost']*ca.sumsqr(self.__vars.get_deviation('imp_stiff'))
            J_total += self.mpc_params['K_cost']*ca.sumsqr(self.__vars['imp_stiff'])

        for mode in self.__modes:
            J_total += self.__params['belief_'+mode]*J[mode] # expected value

        self.__vars.set_x0('imp_stiff', params0['imp_stiff']) # x0 needs to be numerical
        # Set up dictionary of arguments to solve
        w, lbw, ubw, w0 = self.__vars.get_dec_vectors()

        self.args = dict(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        prob = {'f': J_total, 'x': w, 'g': ca.vertcat(*g), 'p': self.__params.get_sym_vec()}

        self.solver = ca.nlpsol('solver', 'ipopt', prob, self.options)

    def build_dec_var_constraints(self):
        ub = {}
        lb = {}

        lb['imp_stiff'] = self.mpc_params['K_min']
        ub['imp_stiff'] = self.mpc_params['K_max']

        return ub, lb


