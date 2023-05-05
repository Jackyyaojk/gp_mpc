# Python includes
from copy import deepcopy

# Library includes
import numpy as np
import casadi as ca

# Custom includes
from .gp_model import GP
from .helper_fns import yaml_load, constraints, compliance_to_world
from .decision_vars import decision_var_set, param_set, decision_var
from autodiff_dynamic_systems.autodiff_sys import Sys

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

        # Set up problem dimensions
        self.__N = mpc_params['mpc_pts']                  # number of MPC steps
        self.__dt = mpc_params['dt']                      # sample time
        self.__N_x = (2+2*mpc_params['state_cov'])*N_p    # number of states of ode
        self.__N_p = N_p
        self.__constraint_slack = mpc_params['constraint_slack']


        self.__gp_dynamics = gp_dynamics_dict
        self.__modes = gp_dynamics_dict.keys()            # names of modes
        #self.__F_int = {mode:gp_dynamics_dict[mode].MDS_system().map(self.__N, 'serial') for mode in self.__modes}
        self.__F_int = {mode:gp_dynamics_dict[mode].MDS_system() for mode in self.__modes}
    
        self.options = yaml_load(path, 'ipopt_params.yaml')
        #jit_options = {"flags": ["-Os"], "verbose": True} # JIT options not found to help much
        #options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        #self.options.update(options)

    def solve(self, params):
        # params are the numerical values of initial robot pose, mode belief, and impedance parameters

        #Create problem and solver
        if not hasattr(self, "solver"):
            self.build_solver(params)

        # Update parameters for the solver
        self.args['p'] = ca.vertcat(*[params[el] for el in params.keys()])
        
        # Solve the NLP
        sol = self.solver(**self.args)

        # Save solution + lagrangian for warm start
        self.args['x0'] = sol['x']
        self.args['lam_x0'] = sol['lam_x']
        self.args['lam_g0'] = sol['lam_g']

        #print(sol['x'])
        self.__vars.set_results(sol['x'])
            
        return self.__vars.filter()
        
    # Formulate the NLP for multiple-shooting
    def build_solver(self, params): 
        N_x = self.__N_x
        N_u = self.__N_p
        ty = ca.MX #if self.mpc_params['precomp'] else ca.SX # Type to use for MPC problem
           # MX has smaller memory footprint, SX is faster.  MX helps alot when using autogen C code.

        # Initialize empty NLP
        J = {mode:0.0 for mode in self.__modes}  # Objective function
        g = []    # constraints functions
        lbg = []  # lower bound on constraints
        ubg = []  # upper-bound on constraints
        vars = {} # decision variables

        # Symbolic varaibles for parameters, these get assigned to numerical values in solve()
        params_sym = param_set(params, symb_type = ty.sym)

        # Build decision variables
        imp_mass = self.mpc_params['imp_mass']*np.ones(3)
        vars['imp_stiff'] = params_sym['imp_stiff']
        vars['des_pose'] = params_sym['pose'][:3]
        for m in self.__modes: vars['x_'+m] = np.zeros((N_x, self.__N-1))
        ub, lb = self.build_dec_var_constraints()
        
        # Turn decision variables into a dec_var object
        self.__vars = decision_var_set(x0 = vars, ub = ub, lb = lb, symb_type = ty.sym)
        
        if self.mpc_params['opti_MBK']:
            g += [self.__vars.get_deviation('imp_stiff')]
            lbg += [-self.mpc_params['delta_K_max']]*self.__N_p
            ubg += [self.mpc_params['delta_K_max']]*self.__N_p
            g += [self.__vars.get_deviation('des_pose')]
            lbg += [-self.mpc_params['delta_xd_max']]*self.__N_p
            ubg += [self.mpc_params['delta_xd_max']]*self.__N_p

        for mode in self.__modes:

            print(imp_mass)
                  
            Fk_next = self.__F_int[mode](x = ca.horzcat(np.zeros((N_x, 1)), self.__vars['x_'+mode]),
                                         des_pose = self.__vars['des_pose'],
                                         init_pose = params_sym['pose'],
                                         imp_mass = imp_mass,
                                         imp_damp = 2*ca.sqrt(vars['imp_stiff']),
                                         imp_stiff = self.__vars['imp_stiff'])
            Xk_next = Fk_next['xf']
            
            J[mode] += ca.sum2(Fk_next['st_cost'])

            g += [ca.reshape(Xk_next[:,:-1]-self.__vars['x_'+mode][:,:], N_x*(self.__N-1), 1)]
            lbg += [ self.__constraint_slack]*N_x*(self.__N-1)
            ubg += [-self.__constraint_slack]*N_x*(self.__N-1)

        # Calculate total objective
        J_total = 0.0
        J_u_total = self.mpc_params['R']*ca.sumsqr(self.__vars.get_deviation('des_pose'))
        if self.mpc_params['opti_MBK']:
            J_u_total += self.mpc_params['delta_K_cost']*ca.sumsqr(self.__vars.get_deviation('imp_stiff'))
            J_u_total += self.mpc_params['K_cost']*ca.sumsqr(self.__vars['imp_stiff'])

        J_total = J_u_total
        for mode in self.__modes:
            J_total += params_sym['belief_'+mode]*J[mode] # expected value

        # Set up dictionary of arguments to solve
        w, lbw, ubw = self.__vars.get_dec_vectors()
        self.__vars.set_x0('imp_stiff', params['imp_stiff'])
        self.__vars.set_x0('des_pose', params['pose'][:3])
        w0 = self.__vars.get_x0()
    
        self.args = dict(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        
        prob = {'f': J_total, 'x': w, 'g': ca.vertcat(*g), 'p': params_sym.get_vector()}
        
        self.solver = ca.nlpsol('solver', 'ipopt', prob, self.options)

    def build_dec_var_constraints(self):
        ub = {}
        lb = {}

        lb['imp_stiff'] = self.mpc_params['K_min']
        ub['imp_stiff'] = self.mpc_params['K_max']
            
        return ub, lb


    def plot(self, plot=True, saveplot=False, Plt_file_Name = 'MPC_trajectories'):
        import matplotlib.pyplot as plt
        T = self.__N*self.__dt
        N = self.__N
        N_x = self.__N_x
        N_p = self.__N_p
        nx2 = int(N_x/2)
        nx4 = int(N_x/4)

        # Plot the solution
        u_plot =   self.__u_traj
        tgrid = [T / N * k for k in range(N + 1)]

        import matplotlib.pyplot as plt
        for mode in self.__modes:
            x_plot =   self.x_traj[mode]
            plt.figure()
            plt.clf()
            colors = 'rgb'
            plt.subplot(2,1,1)
            for ix in range(int(N_x/4)):
                c = colors[ix]
                plt.plot(tgrid, x_plot[ix,:], '--', color = c, label = 'x'+str(1+ix))
                plt.fill_between(tgrid, x_plot[ix]+x_plot[ix+nx2],
                                 x_plot[ix]-x_plot[ix+nx2], color = c, alpha = 0.5)
                plt.plot(tgrid, x_plot[ix+nx4,:], ':',
                         color = c, label = 'x_dot'+str(1+ix))
                plt.fill_between(tgrid, x_plot[ix+nx4]+x_plot[ix+nx2+nx4],

                                 x_plot[ix+nx4]-x_plot[ix+nx2+nx4], color = c, alpha = 0.3)
                plt.legend()
                plt.grid()
                plt.title('Optimal Traj for Mode '+mode)

            plt.subplot(2,1,2)
            for iu in range(N_p):
                plt.step(tgrid, np.append(np.nan, u_plot[iu,:]), '.-',
                         color = colors[iu], label = 'u'+str(1+iu))
                plt.xlabel('t')
                plt.legend()
                plt.grid()
            if saveplot==True: plt.savefig(Plt_file_Name)
            if plot == True: plt.show()

        print("Optimal Delta M {} is".format(u_plot[1 * N_p: 2 * N_p, -1]))
        print("Optimal Delta B {} is".format(u_plot[2 * N_p: 3 * N_p, -1]))
        print("Optimal Delta K {} is".format(u_plot[3 * N_p:, -1]))





