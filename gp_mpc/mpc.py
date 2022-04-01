# Python includes
from copy import deepcopy

# Library includes
import numpy as np
import casadi as ca

# Custom includes
from .gp_model import GP
from .helper_fns import yaml_load, constraints

class MPC:
    def __init__(self, N_p, mpc_params, gp_dynamics_dict):
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

        # l:lower / u:upper bound on all decision variables
        self.__lbd, self.__ubd = self.constraints()

        self.__constraint_slack = mpc_params['constraint_slack']
        self.__precomp = mpc_params['precomp']

        self.options = yaml_load('config/', 'ipopt_params.yaml')
        #jit_options = {"flags": ["-Os"], "verbose": True}
        #options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        #self.options.update(options)

    def solve(self, params):
        # params are the numerical values of initial robot pose, mode belief, and impedance parameters

        #Create problem and solver
        if not hasattr(self, "solver"): self.solverhelper()

        # Update parameters for the solver
        self.args["p"] = params.get_x0()

        # Solve the NLP
        sol = self.solver(**self.args)

        # Save solution + lagrangian for warm start
        self.args['x0'] = sol['x']
        self.args['lam_x0'] = sol['lam_x']
        self.args['lam_g0'] = sol['lam_g']

        self.__x_opt.sol['x']

        self.x_traj = {self.__dec_vars['x_'+m] for m in self.__modes}
        self.u_traj = self.__dec_vars['u']
        self.mbk_traj = self.__dec_vars['imp_params']

        return self.u_traj

    def build_solver(self): # Formulate the NLP for multiple-shooting
        N_x = self.__N_x
        N_u = self.__N_p
        ty = ca.MX if self.mpc_params['precomp'] else ca.SX # Type to use for MPC problem
           # MX has smaller memory footprint, SX is faster.  MX helps alot when using autogen C code.

        # Symbolic varaibles for parameters, these get assigned to numerical values in solve()
        params_initializer = {mode:ty.sym('belief_'+mode,1) for mode in self.__modes}
        params_initializer = {'init_pose': np.zeros(6),
                              'imp_mass': np.zeros(self.__N_p),
                              'imp_damp': np.zeros(self.__N_p),
                              'imp_stiff': np.zeros(self.__N_p),
        }
        params = decision_var_set(ty, params_initializer)

        # Initialize empty NLP
        J = {mode:0.0 for mode in self.__modes}     # Objective function
        g = []    # Constraints function at the optimal solution (ng x 1)
        lbg = []
        ubg = []

        vars = {}

        # Adding shoulder delta
        if self.mpc_params['opti_hum_shoulder']:
            vars['shoulder_pos']  = np.array(self.mpc_params['human_kin']['center'])
        else:
            shoulder_pos = self.mpc_params['human_kin']['center']

        # Adding human joints
        if self.mpc_params['opti_hum_jts']:
            vars['hum_jts'] = np.zeros(4)

            hum_wrist_pos, _  = self.__hum_FK(jts, shoulder_pos)
            g += [init_pose[:3]-hum_wrist_pos]
            lbg += list(np.zeros(3))
            ubg += list(np.zeros(3))

        # Impedance
        if self.mpc_params['opti_MBK']:
            vars['imp_mass'] = params['imp_mass'] # initial value
            vars['imp_damp'] = params['imp_damp'] # initial value

            lbg += [-self.mpc_params['delta_M_max']]*self.__N_p
            lbg += [-self.mpc_params['delta_B_max']]*self.__N_p
            ubg += [self.mpc_params['delta_M_max']]*self.__N_p
            ubg += [self.mpc_params['delta_B_max']]*self.__N_p

        for m in self.__modes: vars[mode] = np.zeros((N_x, self.__N-1))
        vars['u'] = np.zeros((N_u, self.__N))

        ub, lb = self.build_dec_var_constraints()
        self.__vars = decision_var_set(x0 = vars, ub = ub, lb = lb)

        for mode in self.__modes:
            Fk_next = self.__F_int[mode](x = ca.horzcat(np.zeros((N_x, 1)), self.__vars[mode]),
                                         u = self.__vars['u'],
                                         init_pose = params['init_pose'],
                                         hum_kin_opti = ca.vertcat(params['hum_shoulder_opt'],
                                                                   params['hum_joint_opt']),
                                         imp_params = params['imp_params'])

### REWRITTEN TO HERE
            Xk_next = Fk_next['xf']

            J[mode] += ca.sum2(Fk_next['st_cost'])
            w += [ca.reshape(Xk[mode][:,1:], N_x*(self.__N-1), 1)]
            lbw += list(self.__lbx)*(self.__N-1)
            ubw += list(self.__ubx)*(self.__N-1)
            w0 += list(np.zeros((N_x*(self.__N-1))))
            g += [ca.reshape(Xk_next[:,:-1]-Xk[mode][:,1:], N_x*(self.__N-1), 1)]
            lbg += [ self.__constraint_slack]*N_x*(self.__N-1)*len(self.__modes)
            ubg += [-self.__constraint_slack]*N_x*(self.__N-1)*len(self.__modes)

            if self.mpc_params['chance_prob'] != 0.0:
                # Adding chance constraints for force
                chance_const = self.mpc_params['chance_bnd'] - Fk_next['hum_force_cart'][-1,2]
                chance_const -= ca.erfinv(self.mpc_params['chance_prob'])*ca.sqrt(Fk_next['f_cov'][-1])
                print("Adding chance constraints")
                g += [chance_const.T]
                lbg += list(np.zeros(1))
                ubg += list(np.full(1, np.inf))

            if self.mpc_params['well_damped_margin'] != 0.0:
                print('Adding well-damped constraint')
                Ke = ca.fabs(self.__gp_dynamics[mode].gp_grad(Xk[mode][:self.__N_p,-1]))[2]
                g += [imp_params[self.__N_p+2]-4*imp_params[2]*Ke-self.mpc_params['well_damped_margin']]
                lbg += list(np.zeros(1))
                ubg += list(np.full(1, np.inf))

            x_opt[mode] += [Xk[mode]]
            hum_force_opt += [Fk_next['hum_force_cart']]
            hum_jt_torque_opt += [Fk_next['hum_force_joint']]


        # Calculate total objective
        J_total = 0.0
        J_u_total = 0.0
        for U in u_opt:
            J_u_total += self.mpc_params['R']*ca.sumsqr(U)
        if self.mpc_params['opti_MBK']:
            J_u_total += self.mpc_params['delta_M_cost']*ca.sumsqr(imp_params_delta[:self.__N_p])
            J_u_total += self.mpc_params['delta_B_cost']*ca.sumsqr(imp_params_delta[self.__N_p:])

        if self.mpc_params['risk_sens'] == 0: # Expected cost
            J_total = J_u_total
            for mode in self.__modes:
                J_total += belief[mode]*J[mode] # expected value
        else: # Risk-sensitive formulation. See, e.g. Medina2012
            for mode in self.__modes:
                J_total += belief[mode]*ca.exp(-0.5*self.mpc_params['risk_sens']*(J[mode]))
            J_total = -2/self.mpc_params['risk_sens']*ca.log(J_total)+J_u_total

        # Build parameter list
        p = [init_pose]
        for mode in self.__modes:
            p += [belief[mode]]
        p += [init_imp_params]
#        p += [imp_par_bnd]

        # Functions to get x, u and human force from w
        self.extract_traj = { mode: ca.Function('extract_traj', [ca.vertcat(*w)],
                                                  [Xk[mode]],
                                                  ['w'], ['x_'+mode])\
                              for mode in self.__modes }
        self.extract_ctrl = ca.Function('extract_ctrl', [ca.vertcat(*w)],
                                                  [ca.horzcat(*u_opt)],
                                                  ['w'], ['u'])
        self.extract_hum = ca.Function('extract_hum', [ca.vertcat(*w), init_pose],
                                       [ca.horzcat(*hum_force_opt), ca.horzcat(*hum_jt_torque_opt),
                                        hum_shoulder_opt, hum_joint_opt],
                                       ['w', 'init_pose'], ['hum_force', 'hum_joint_torque', 'hum_shoulder', 'hum_joints'])
        imp_params_to_return = imp_params_opt[0] if len(imp_params_opt)>0 else []
        self.extract_mbk = ca.Function('extract_mbk', [ca.vertcat(*w), init_imp_params], [imp_params_to_return])
        self.extract_f_cov = ca.Function('extract_f_cov', [ca.vertcat(*w), init_imp_params, init_pose], [Fk_next['f_cov']])
        # Set up dictionary of arguments to solve
        self.args = dict(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        prob = {'f': J_total, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p': ca.vertcat(*p)}
        if not self.__precomp:
            self.solver = ca.nlpsol('solver', 'ipopt', prob, self.options)
            #self.solver = ca.nlpsol('solver', 'blocksqp', prob, {})
        else:
            import subprocess
            import time
            #gen_opts = {}
            gen_opts={'casadi_int': 'long int'} #'casadi_real': 'float' # requires special IPOPT compile flag
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
        lb['u'] = self.__lbu
        ub['u'] = self.__ubu
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





