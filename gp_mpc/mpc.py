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
    def __init__(self, N_p, mpc_params, gp_dynamics_dict, path):
        self.mpc_params = mpc_params

        # Set up problem dimensions
        self.__N = mpc_params['mpc_pts']                  # number of MPC steps
        self.__dt = mpc_params['dt']                      # sample time
        self.__N_x = (2+2*mpc_params['state_cov'])*N_p    # number of states of ode
        self.__N_p = N_p

        self.__gp_dynamics = gp_dynamics_dict
        self.__modes = gp_dynamics_dict.keys()
        self.__F_int = {mode:gp_dynamics_dict[mode].MDS_system().map(self.__N, 'serial') for mode in self.__modes}
        self.__hum_FK = list(gp_dynamics_dict.values())[0].human_FK

        # l:lower / u:upper bound on all decision variables, now built from dec_var object
        # self.__lbd, self.__ubd = self.build_dec_var_constraints()

        self.__constraint_slack = mpc_params['constraint_slack']
        self.__precomp = mpc_params['precomp']

        self.options = yaml_load(path, 'ipopt_params.yaml')
        #jit_options = {"flags": ["-Os"], "verbose": True}
        #options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        #self.options.update(options)

    def solve(self, params):
        # params are the numerical values of initial robot pose, mode belief, and impedance parameters

        #Create problem and solver
        if not hasattr(self, "solver"): self.build_solver(params)

        # Update parameters for the solver
        self.args['p'] = ca.vertcat(*[params[el] for el in params.keys()])
        # Solve the NLP
        sol = self.solver(**self.args)

        # Save solution + lagrangian for warm start
        self.args['x0'] = sol['x']
        self.args['lam_x0'] = sol['lam_x']
        self.args['lam_g0'] = sol['lam_g']

        self.__vars.set_results(sol['x'])
        self.x_traj = {m:self.__vars['x_'+m] for m in self.__modes}
       #print(self.x_traj)
        self.u_traj = self.__vars['u']
        self.imp_mass = np.squeeze(self.__vars['imp_mass'])
        self.imp_damp = np.squeeze(self.__vars['imp_damp'])

        return self.u_traj

    def build_solver(self, params): # Formulate the NLP for multiple-shooting
        N_x = self.__N_x
        N_u = self.__N_p
        ty = ca.MX #if self.mpc_params['precomp'] else ca.SX # Type to use for MPC problem
           # MX has smaller memory footprint, SX is faster.  MX helps alot when using autogen C code.
        # Symbolic varaibles for parameters, these get assigned to numerical values in solve()
        params = param_set(params, symb_type = ty.sym)

        # Initialize empty NLP
        J = {mode:0.0 for mode in self.__modes}     # Objective function
        g = []    # Constraints function at the optimal solution (ng x 1)
        lbg = []
        ubg = []

        vars = {}

        # Impedance
        if self.mpc_params['opti_MBK']:
            vars['imp_mass'] = params['imp_mass']
            vars['imp_damp'] = params['imp_damp']

        for m in self.__modes: vars['x_'+m] = np.zeros((N_x, self.__N-1))
        vars['u'] = np.zeros((N_u, self.__N))

        ub, lb = self.build_dec_var_constraints()
        self.__vars = decision_var_set(x0 = vars, ub = ub, lb = lb, symb_type = ty.sym)

        if self.mpc_params['opti_MBK']:
            g += [self.__vars.get_deviation('imp_mass')]
            g += [self.__vars.get_deviation('imp_damp')]
            lbg += [-self.mpc_params['delta_M_max']]*self.__N_p
            lbg += [-self.mpc_params['delta_B_max']]*self.__N_p
            ubg += [self.mpc_params['delta_M_max']]*self.__N_p
            ubg += [self.mpc_params['delta_B_max']]*self.__N_p

        for mode in self.__modes:
            Fk_next = self.__F_int[mode](x = ca.horzcat(np.zeros((N_x, 1)), self.__vars['x_'+mode]),
                                         u = self.__vars['u'],
                                         init_pose = params['init_pose'],
                                         imp_mass = self.__vars['imp_mass'],
                                         imp_damp = self.__vars['imp_damp'])
            Xk_next = Fk_next['xf']
            J[mode] += ca.sum2(Fk_next['st_cost'])

            g += [ca.reshape(Xk_next[:,:-1]-self.__vars['x_'+mode][:,:], N_x*(self.__N-1), 1)]
            lbg += [ self.__constraint_slack]*N_x*(self.__N-1)*len(self.__modes)
            ubg += [-self.__constraint_slack]*N_x*(self.__N-1)*len(self.__modes)

            if self.mpc_params['chance_prob'] != 0.0:
                # Adding chance constraints for force
                chance_const = self.mpc_params['chance_bnd'] - Fk_next['hum_force_cart'][2, -1]# [-1,2]
                chance_const -= ca.erfinv(self.mpc_params['chance_prob'])*ca.sqrt(Fk_next['f_cov'][-1])
                g += [chance_const.T]
                chance_const = self.mpc_params['chance_bnd'] - Fk_next['hum_force_cart'][2, 0]# [-1,2]
                chance_const -= ca.erfinv(self.mpc_params['chance_prob'])*ca.sqrt(Fk_next['f_cov'][-1])
                print("Adding chance constraints")
                g += [chance_const.T]
                lbg += list(np.zeros(2))
                ubg += list(np.full(2, np.inf))

            if self.mpc_params['well_damped_margin'] != 0.0:
                print('Adding well-damped constraint')
                x = self.__vars['x_'+mode][:self.__N_p,0]
                x += ca.DM([0, 0, 0.03])
                #x = self.__vars['x_'+mode][3:3+self.__N_p,0]
                #x = Xk_next[:self.__N_p,-1]
                #x = 10*self.mpc_params['dt']*self.__vars['x_'+mode][3:3+self.__N_p,0]

                x_w = compliance_to_world(params['init_pose'],x)
                Ke = ca.MX(ca.fabs(self.__gp_dynamics[mode].gp_grad(x_w))[2,2])
                #g += [self.__vars['imp_damp'][2]-2*ca.sqrt(self.__vars['imp_mass'][2]*(Ke))*self.mpc_params['well_damped_margin']]
                g += [1e-2*(self.__vars['imp_damp'][2]*700-2*(self.__vars['imp_mass'][2]*(Ke))*self.mpc_params['well_damped_margin'])]
                lbg += list(np.zeros(1))
                ubg += list(np.full(1, np.inf))

        # Calculate total objective
        J_total = 0.0
        J_u_total = self.mpc_params['R']*ca.sumsqr(self.__vars['u'])
        if self.mpc_params['match_force_setpoint']:
            J_u_total += self.mpc_params['match_force_setpoint_weight']*\
                ca.sumsqr(ca.fmax(self.mpc_params['match_force_setpoint']-self.__vars['u'][2,:], 0))
        if self.mpc_params['opti_MBK']:
            J_u_total += self.mpc_params['delta_M_cost']*ca.sumsqr(self.__vars.get_deviation('imp_mass'))
            J_u_total += self.mpc_params['delta_B_cost']*ca.sumsqr(self.__vars.get_deviation('imp_damp'))
            J_u_total += self.mpc_params['M_cost']*ca.sumsqr(self.__vars['imp_mass'])
            J_u_total += self.mpc_params['B_cost']*ca.sumsqr(self.__vars['imp_damp'])

        if self.mpc_params['risk_sens'] == 0: # Expected cost
            J_total = J_u_total
            for mode in self.__modes:
                J_total += params['belief_'+mode]*J[mode] # expected value
        else: # Risk-sensitive formulation. See, e.g. Medina2012
            for mode in self.__modes:
                J_total += belief[mode]*ca.exp(-0.5*self.mpc_params['risk_sens']*(J[mode]))
            J_total = -2/self.mpc_params['risk_sens']*ca.log(J_total)+J_u_total

        if self.mpc_params['dist_rej']:
            admittance_TF1 = Sys([1],
                                [self.__vars['imp_mass'][1],
                                 self.__vars['imp_damp'][1]],
                                symb_type = ty)
            admittance_TF2 = Sys([1],
                                [self.__vars['imp_mass'][2],
                                 self.__vars['imp_damp'][2]],
                                symb_type = ty)
            force_signal = Sys([1, 0],[1, self.mpc_params['dist_omega']])
            self.dist_signal1 = admittance_TF1*force_signal
            self.dist_signal2 = admittance_TF2*force_signal
            J_total += self.mpc_params['dist_rej']*self.dist_signal1.h2(sol = 'lapackqr')
            J_total += self.mpc_params['dist_rej']*self.dist_signal2.h2(sol = 'lapackqr') #solver: scipy, ma27, lapackqr, ldl. Probably just need scipy if poorly conditioned, otherwise the other solvers are casadi native  and faster.

        # Set up dictionary of arguments to solve
        w, lbw, ubw = self.__vars.get_dec_vectors()
        w0 = self.__vars.get_x0()

        self.args = dict(x0=np.zeros(w.shape), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        prob = {'f': J_total, 'x': w, 'g': ca.vertcat(*g), 'p': params.get_vector()}
        if not self.__precomp:
            self.solver = ca.nlpsol('solver', 'ipopt', prob, self.options)
            #self.solver = ca.nlpsol('solver', 'blocksqp', prob, {})
        else:
            import subprocess
            import time
            gen_opts = {}
            solver = ca.nlpsol('solver', 'ipopt', prob, self.options)
            solver.generate_dependencies("nlp.c", gen_opts)
            start = time.time()
            subprocess.Popen("gcc -fPIC -shared -O1 nlp.c -o nlp.so", shell=True).wait()
            print("Compile time was: {}".format(time.time()-start))
            self.solver = ca.nlpsol("solver", "ipopt", "./nlp.so", self.options)

    def build_dec_var_constraints(self):
        ub = {}
        lb = {}

        lb['shoulder_pos'] = np.array(self.mpc_params['human_kin']['center'])-self.mpc_params['max_shoulder']*np.array([1,1,0])
        ub['shoulder_pos'] = np.array(self.mpc_params['human_kin']['center'])+self.mpc_params['max_shoulder']*np.array([1,1,0])
        lb['hum_jts'] = np.array(self.mpc_params['hum_jt_lim']['low'])*np.pi
        ub['hum_jts'] = np.array(self.mpc_params['hum_jt_lim']['high'])*np.pi
        lb['imp_mass'] = self.mpc_params['M_min']
        lb['imp_damp'] = self.mpc_params['B_min']
        ub['imp_mass'] = self.mpc_params['M_max']
        ub['imp_damp'] = self.mpc_params['B_max']
        if self.__N_p == 3:
            lb['u'] = -self.mpc_params['u_lin_max']
            ub['u'] = self.mpc_params['u_lin_max']
        else:
            lb['u'] = np.repeat(np.expand_dims(np.concatenate([[-self.mpc_params['u_lin_max']]*3, [-self.mpc_params['u_rot_max']]*3]),axis=1), self.__N, axis=1)
            ub['u'] = np.repeat(np.expand_dims(np.concatenate([[self.mpc_params['u_lin_max']]*3,  [self.mpc_params['u_rot_max']]*3]),axis=1), self.__N, axis=1)
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





