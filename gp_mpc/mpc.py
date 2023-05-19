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
      N_p: dim of positions in dynamics
      mpc_params: the dict for the MPC parameters
      gp_dynamics_dict: the dict of different dynamics objects 
    '''
    def __init__(self, mpc_params, gp_dynamics_dict, path):
        self.mpc_params = mpc_params
        self.options = yaml_load(path, 'ipopt_params.yaml')
        
        # Set up problem dimensions
        self.__N_p = list(gp_dynamics_dict.values())[0].state_dim 
        self.__N = mpc_params['mpc_pts']                      # number of MPC steps
        self.__N_x = (2+2*mpc_params['state_cov'])*self.__N_p # number of states of ode
        
    

        self.__gp_dynamics = gp_dynamics_dict
        self.__modes = gp_dynamics_dict.keys()            # names of modes
        #self.__dyn = {mode:gp_dynamics_dict[mode].build_dynamics().map(self.__N, 'serial') for mode in self.__modes}
        self.__dyn = {mode:gp_dynamics_dict[mode].build_dynamics() for mode in self.__modes}
        
        
    def solve(self, pars):
        # IN: params are the numerical values of initial robot pose, mode belief, and impedance parameters

        #Create problem and solver
        if not hasattr(self, "solver"):
            self.build_solver(pars)

        # Update parameters for the solver
        self.args['p'] = self.__pars.update(pars)

        # Solve the NLP
        sol = self.solver(**self.args)

        # Save solution + lagrangian for warm start
        self.args['x0'] = sol['x']
        self.args['lam_x0'] = sol['lam_x']
        self.args['lam_g0'] = sol['lam_g']

        self.__vars.set_results(sol['x'])
        return self.__vars.get_dec_dict()

    # Formulate the NLP for multiple-shooting
    def build_solver(self, pars0): 
        N_x = self.__N_x
        N_p = self.__N_p

        # Initialize empty NLP
        J = 0      # objective function
        vars0 = {} # initial values for decision variables

        # Symbolic varaibles for parameters, these get assigned to numerical values in solve()
        self.__pars = ParamSet(pars0)

        # Build decision variables
        vars0['imp_stiff'] = self.__pars['imp_stiff']   # imp stiff in tcp coord, initial value is current stiff
        vars0['des_pose'] = np.zeros((N_p))             # rest position of imp spring relative to tcp
        for m in self.__modes:
            vars0['x_'+m] = np.zeros((N_x, self.__N-1)) # trajectory relative to tcp
        ub, lb = self.build_dec_var_constraints()
        self.__vars = DecisionVarSet(x0 = vars0, ub = ub, lb = lb)

        self.build_constraints()
        
        for mode in self.__modes:
            dyn_next = self.__dyn[mode](x = ca.horzcat(np.zeros((N_x, 1)), self.__vars['x_'+mode]),
                                        des_pose = self.__vars['des_pose'],
                                        init_pose = self.__pars['pose'],
                                        imp_mass = self.mpc_params['imp_mass']*np.ones(3),
                                        imp_damp = 2*ca.sqrt(self.__vars['imp_stiff']),
                                        imp_stiff = self.__vars['imp_stiff'])
    
            self.add_continuity_constraints(dyn_next['x_next'], self.__vars['x_'+mode])        
            J += self.__pars['belief_'+mode]*ca.sum2(dyn_next['st_cost'])

        # Add control costs
        J += self.mpc_params['delta_xd_cost']*ca.sumsqr(self.__vars['des_pose'])
        if self.mpc_params['opti_MBK']:
            J += self.mpc_params['delta_K_cost']*ca.sumsqr(self.__vars.get_deviation('imp_stiff'))
            J += self.mpc_params['K_cost']*ca.sumsqr(self.__vars['imp_stiff'])

        # Need x0 to be numerical when building solver
        self.__vars.set_x0('imp_stiff', pars0['imp_stiff'])

        # Set up dictionary of arguments to solve
        x, lbx, ubx, x0 = self.__vars.get_dec_vectors()

        self.args = dict(x0=x0, lbx=lbx, ubx=ubx, lbg=self.__lbg, ubg=self.__ubg)
        prob = dict(f=J, x=x, g=ca.vertcat(*self.__g), p=self.__pars.get_sym_vec())
        self.solver = ca.nlpsol('solver', 'ipopt', prob, self.options)

    def build_constraints(self):
        # General NLP constraints, not including continuity constraints
        self.__g = []      # constraints functions
        self.__lbg = []    # lower bound on constraints
        self.__ubg = []    # upper-bound on constraints
        if self.mpc_params['opti_MBK']:
            self.__g += [self.__vars.get_deviation('imp_stiff')]
            self.__lbg += [-self.mpc_params['delta_K_max']]*self.__N_p
            self.__ubg += [self.mpc_params['delta_K_max']]*self.__N_p
            self.__g += [self.__vars['des_pose']]
            self.__lbg += [-self.mpc_params['delta_xd_max']]*self.__N_p
            self.__ubg += [self.mpc_params['delta_xd_max']]*self.__N_p

    def add_continuity_constraints(self, x_next, x):
        N_x = self.__N_x
        N = self.__N
        self.__g += [ca.reshape(x_next[:,:-1]-x, N_x*(N-1), 1)]
        self.__lbg += [ self.mpc_params['constraint_slack']]*N_x*(N-1)
        self.__ubg += [-self.mpc_params['constraint_slack']]*N_x*(N-1)
        
    def build_dec_var_constraints(self):
        ub = {}
        lb = {}
        lb['imp_stiff'] = self.mpc_params['K_min']
        ub['imp_stiff'] = self.mpc_params['K_max']

        return ub, lb


