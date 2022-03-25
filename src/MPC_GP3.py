# Python includes
from copy import deepcopy

# Library includes
import numpy as np
import casadi as ca

# Custom includes
from gp_mpc import GP
from helper_fns import yaml_load, constraints

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

        # l:lower / u:upper bound on x:control/u:input
        self.__lbx, self.__ubx, self.__lbu, self.__ubu =\
              constraints(N_p, mpc_params)

        self.__constraint_slack = mpc_params['constraint_slack']
        self.__precomp = mpc_params['precomp']

        self.options = yaml_load('config/ipopt_params.yaml')
        #jit_options = {"flags": ["-Os"], "verbose": True}
        #options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
        #self.options.update(options)

        if mpc_params['print_IPOPT']:
            self.pipe_path = "logs/mpc_log"
            with open(self.pipe_path, "w") as p:
                p.truncate(0)
                p.write("***This is the IPOPT DEBUG WINDOW***\n  ~~~Prepare for amazing detail~~~ \n")
            print("Opened log on {}".format(self.pipe_path))
            self.out_file_path = "logs/ipopt_print"
            self.options['ipopt.output_file'] = self.out_file_path

    def solve(self, init_pose_num = None, belief_num = None, imp_params_num_mass = None, imp_params_num_damp = None):
        # arguments are the numerical values of initial robot pose, mode belief, and impedance parameters
        # Set initial conditions:
        if init_pose_num.any() == None:
            print('No init pose given to MPC solve')
            init_pose_num = np.zeros(6)
        if belief_num == None:
            print('No belief given to MPC solve')
            belief_num = {mode:1.0/len(self.__modes) for mode in self.__modes}
        if imp_params_num_mass.any() == None:
            print('No initial impedance params given, using defaults')
            imp_params_num_mass = np.array([10., 10., 10., 0.5, 0.5, 0.5])
            imp_params_num_damp = np.array([1000., 1000., 1000., 200., 200., 200])
        imp_params_num = np.hstack((imp_params_num_mass[:self.__N_p], imp_params_num_damp[:self.__N_p]))
        # Build parameter vector p_num, which is the numerical value of all non-decision variables
        p_num = init_pose_num
        for mode in self.__modes:
            p_num = np.append(p_num, belief_num[mode])
        p_num = np.append(p_num, imp_params_num)
        p_num_imp = deepcopy(p_num)
        #p_num = np.append(p_num, 0.4*np.ones(2*self.__N_p)) # bound those pesky imp pars
        #Create problem and solver
        if not hasattr(self, "solver"):
            self.solverhelper()
            #self.solverhelper_imp()

        # Update parameters for the solver (initial state, etc)
        self.args["p"] = p_num

        # Solve the NLP
        sol = self.solver(**self.args)

        # Save solution + lagrangian for warm start
        self.args['x0'] = sol['x']
        self.args['lam_x0'] = sol['lam_x']
        self.args['lam_g0'] = sol['lam_g']

        # Extract components of solution
        self.x_traj = { mode: self.extract_traj[mode](sol['x']).full()\
                        for mode in self.__modes }
        self.u_traj = self.extract_ctrl(sol['x']).full()
        if self.mpc_params['opti_hum_shoulder'] or self.mpc_params['opti_hum_jts']:
            hum_force, hum_jt_torque, hum_shoulder, hum_joints = self.extract_hum(sol['x'], init_pose_num)
        if self.mpc_params['opti_hum_shoulder']: self.hum_shoulder = np.squeeze(hum_shoulder.full())
        if self.mpc_params['opti_hum_jts']: self.hum_joints = np.squeeze(hum_joints.full())

        x_traj_stacked = np.vstack([self.x_traj[mode][:,1:] for mode in self.__modes])
        x_traj_stacked = np.append(x_traj_stacked, self.u_traj)
        p_num_imp = np.append(x_traj_stacked, p_num_imp)
        #self.args_imp["p"] = p_num_imp
        #sol_imp = self.solver_imp(**self.args_imp) #TODO: test on dev laptop
        if self.mpc_params['opti_MBK']: self.mbk_traj = np.squeeze(self.extract_mbk(sol['x'],imp_params_num).full())
                                                       #self.extract_mbk(sol_imp['x'], imp_params_num)
        # Write the debug logs
        if hasattr(self, "pipe_path"):
            try:
                with open(self.pipe_path, "w") as p, open(self.out_file_path) as ip:
                    for line in ip:
                        p.write(line)
                    p.write("\ninitial state:  \n{}\n".format(init_pose_num))
                    p.write("ctrl soln:      \n{}\n".format(self.u_traj))
                    p.write("state soln:     \n{}\n".format(self.x_traj))
                    for mode in self.__modes:
                        p.write("cost, mode {}: \n{}\n\n\n".\
                                format(mode, self.__gp_dynamics[mode].split_cost_function(\
                                       self.x_traj[mode], self.u_traj)))
            except BrokenPipeError:
                print(BrokenPipeError)
                pass
        return self.u_traj

    def solverhelper(self): # Formulate the NLP for multiple-shooting
        N_x = self.__N_x
        N_u = self.__N_p
        ty = ca.MX if self.mpc_params['precomp'] else ca.SX # Type to use for MPC problem
           # MX has smaller memory footprint, SX is faster.  MX helps alot when using autogen C code.

        # Symbolic varaibles for parameters, these get assigned to numerical values in solve()
        init_pose = ty.sym('init_pose',6)
        belief = {mode:ty.sym('belief_'+mode,1) for mode in self.__modes}
        init_imp_params = ty.sym('init_imp_params',2*self.__N_p)
        imp_par_bnd = ty.sym('imp_par_bnd', 2*self.__N_p)

        # Initialize empty NLP
        w = []    # Decision variables at the optimal solution ((num_modes x N_xc) x 1)
        w0 = []   # Decision variables, initial guess (nxc x 1)
        lbw = []  # Decision variables lower bound (nxc x 1), default -inf.
        ubw = []  # Decision variables upper bound (nxc x 1), default +inf.
        J = {mode:0.0 for mode in self.__modes}     # Objective function
        g = []    # Constraints function at the optimal solution (ng x 1)
        lbg = []
        ubg = []

        # Lists to track collections of decision variables
        x_opt = {mode:[] for mode in self.__modes}
        u_opt = []             # optimized robot decision variables
        imp_params_opt = []    # impedance parameters
        hum_force_opt = []     # human forces (lin)
        hum_jt_torque_opt = [] # human joint torques
        hum_shoulder_opt = []  # human shoulder position
        hum_joint_opt = []     # human joint angles

        # Adding shoulder delta
        if self.mpc_params['opti_hum_shoulder']:
            shoulder_pos = ty.sym('shoulder_pos',3)
            w   += [shoulder_pos]
            lbw += list(np.array(self.mpc_params['human_kin']['center'])[:2]\
                        -self.mpc_params['max_shoulder'])
            lbw += [self.mpc_params['human_kin']['center'][2]]
            ubw += list(np.array(self.mpc_params['human_kin']['center'])[:2]\
                        +self.mpc_params['max_shoulder'])
            ubw += [self.mpc_params['human_kin']['center'][2]]
            w0  += list(self.mpc_params['human_kin']['center'])
            hum_shoulder_opt = shoulder_pos
        else:
            shoulder_pos = self.mpc_params['human_kin']['center']

        # Adding human joints
        if self.mpc_params['opti_hum_jts']:
            jts  = ty.sym('hum_jts', 4)
            w   += [jts]
            lbw += list(np.array(self.mpc_params['hum_jt_lim']['low'])*np.pi)
            ubw += list(np.array(self.mpc_params['hum_jt_lim']['high'])*np.pi)
            w0  += list(np.zeros(4))
            hum_joint_opt = jts
            hum_wrist_pos, _  = self.__hum_FK(jts, shoulder_pos)
            g += [init_pose[:3]-hum_wrist_pos]
            lbg += list(np.zeros(3))
            ubg += list(np.zeros(3))

        # Impedance
        imp_params = init_imp_params
        if self.mpc_params['opti_MBK']:
            imp_params_delta = ty.sym('delta_imp', 2*self.__N_p)
            imp_params += imp_params_delta
            imp_params_opt += [imp_params_delta]
            w += [imp_params_delta]
            lbw += [-self.mpc_params['delta_M_max']]*self.__N_p
            lbw += [-self.mpc_params['delta_B_max']]*self.__N_p
            ubw += [self.mpc_params['delta_M_max']]*self.__N_p
            ubw += [self.mpc_params['delta_B_max']]*self.__N_p
            w0 += list(np.zeros(2*self.__N_p))

            g += [imp_params]
            lbg += [self.mpc_params['M_min']]*self.__N_p
            lbg += [self.mpc_params['B_min']]*self.__N_p
            ubg += [np.inf]*2*self.__N_p

        # Dynamics!
        Xk = {}
        Uk = ty.sym('uk', N_u, self.__N)
        u_opt += [Uk]
        w += [ca.reshape(Uk, N_u*self.__N, 1)]
        lbw += self.__lbu*self.__N
        ubw += self.__ubu*self.__N
        w0 += list(np.zeros(N_u*self.__N))
        for mode in self.__modes:
            Xk[mode] = ca.horzcat(np.zeros((N_x, 1)),
                                  ty.sym('xk', N_x, (self.__N-1)))

            Fk_next = self.__F_int[mode](x = Xk[mode], u = Uk,
                                         init_pose = init_pose,
                                         hum_kin_opti = ca.vertcat(hum_shoulder_opt, hum_joint_opt),
                                         imp_params = imp_params)
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

    def solverhelper_imp(self): # Formulate the NLP for multiple-shooting
        N_x = self.__N_x
        N_u = self.__N_p
        ty = ca.MX if self.mpc_params['precomp'] else ca.SX # Type to use for MPC problem
           # MX has smaller memory footprint, SX is faster.  MX helps alot when using autogen C code.

        # Symbolic varaibles for parameters, these get assigned to numerical values in solve()
        init_pose = ty.sym('init_pose',6)
        belief = {mode:ty.sym('belief_'+mode,1) for mode in self.__modes}
        init_imp_params = ty.sym('init_imp_params',2*self.__N_p)
        imp_par_bnd = ty.sym('imp_par_bnd', 2*self.__N_p)
        # Initialize empty NLP
        w = []    # Decision variables at the optimal solution ((num_modes x N_xc) x 1)
        w0 = []   # Decision variables, initial guess (nxc x 1)
        lbw = []  # Decision variables lower bound (nxc x 1), default -inf.
        ubw = []  # Decision variables upper bound (nxc x 1), default +inf.
        J = {mode:0.0 for mode in self.__modes}     # Objective function
        g = []    # Constraints function at the optimal solution (ng x 1)
        lbg = []
        ubg = []
        p = []

        # Lists to track collections of decision variables
        imp_params_opt = []    # impedance parameters

        # Impedance
        imp_params = init_imp_params
        
        imp_params_delta = ty.sym('delta_imp', 2*self.__N_p)
        imp_params += imp_params_delta
        imp_params_opt += [imp_params_delta]
        w += [imp_params_delta]
        lbw += [-0.4]*2*self.__N_p
        ubw += [0.4]*2*self.__N_p
        w0 += list(np.zeros(2*self.__N_p))

        # Dynamics!
        Xk = {}
        Uk = ty.sym('uk', N_u, self.__N)
        for mode in self.__modes:
            Xk[mode] = ca.horzcat(np.zeros((N_x, 1)),
                                  ty.sym('xk', N_x, (self.__N-1)))

            Fk_next = self.__F_int[mode](x = Xk[mode], u = Uk,
                                         init_pose = init_pose,
                                         hum_kin_opti = [],
                                         imp_params = imp_params)
            J[mode] += ca.sum2(Fk_next['st_cost'])
            #w += [ca.reshape(Xk[mode][:,1:], N_x*(self.__N-1), 1)]
            #lbw += list(self.__lbx)*(self.__N-1)
            #ubw += list(self.__ubx)*(self.__N-1)
            #w0 += list(np.zeros((N_x*(self.__N-1))))
            #g += [ca.reshape(Xk_next[:,:-1]-Xk[mode][:,1:], N_x*(self.__N-1), 1)]

            p += [ca.reshape(Xk[mode][:,1:], N_x*(self.__N-1), 1)] 
        p += [ca.reshape(Uk, N_u*self.__N, 1)]
        # Calculate total objective
        J_total = 0.0
        J_u_total = 0.0
        J_u_total += self.mpc_params['R']*ca.sumsqr(Uk)
        if self.mpc_params['opti_MBK']:
            J_u_total += 1*ca.sumsqr(imp_params_delta)

        if self.mpc_params['risk_sens'] == 0: # Expected cost
            J_total = J_u_total
            for mode in self.__modes:
                J_total += belief[mode]*J[mode] # expected value
        else: # Risk-sensitive formulation. See, e.g. Medina2012
            for mode in self.__modes:
                J_total += belief[mode]*ca.exp(-0.5*self.mpc_params['risk_sens']*(J[mode]))
            J_total = -2/self.mpc_params['risk_sens']*ca.log(J_total)+J_u_total

        # Build parameter list
        p += [init_pose]
        for mode in self.__modes:
            p += [belief[mode]]
        p += [init_imp_params]
        

        # Functions to get x, u and human force from w
        imp_params_to_return = imp_params_opt[0] if len(imp_params_opt)>0 else []
        self.extract_mbk = ca.Function('extract_mbk', [ca.vertcat(*w), init_imp_params], [imp_params_to_return])
        # Set up dictionary of arguments to solve
        self.args_imp = dict(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # Create an NLP problem
        prob = {'f': J_total, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p': ca.vertcat(*p)}
        self.solver_imp = ca.nlpsol('solver', 'ipopt', prob, self.options)


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





