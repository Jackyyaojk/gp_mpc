# MPC libary
# Includes
import numpy as np
import casadi as ca
from mpl_toolkits.mplot3d import axes3d #for Data generation for GP training
import matplotlib.pyplot as plt

from gp_mpc import Model, GP, MPC, plot_eig, lqr
import itertools

from sys import path
import os
import subprocess
import time
#____________________________________________________________________________________________________________________________________________


class MPC:
    def __init__(self, N_p, horizon, interval,
                 x_sp = None, Q_fin=1, FinCost= False,
                 Save_IPOPT = False, Opti_MBK = False, print_IPOPT = False,
                 mu_strategy = 'probing', Out_file_Name = "IP_Opt_Results",
                 expect_infeasible_problem='no', bound_relax_factor=1e-5, scc = 0,  gp_dynamics = None,
                 lbx = None, ubx = None, lbu = None, ubu = None, precomp = True):
        # Parameter
        # N_x         Number of states
        # N_u         Number of Inputs
        # horizon     MPC window in seconds
        # intervall   MPC steps within horizon
        # x_sp        state that has to be reached -> Implement for multiple modes
        # Q_stage     Stage cost of states
        # R_stage     Stage cost of inputs
        # discrete method:
        #       "MS_??" ->      Multiple shooting with different integrators
        #           MS__rk      using runge kutta integrator from casadi
        #           MS__rk4     using runge kutta integrator from Langaker
        #           MS__cvodes  using CVODES integrator from casadi
        #       "OC"            using Orthogonal colocation with 4 collocation points
        #

        # Todo: Variable/GP cost function for multimode
        self.__FinCost = FinCost
        self.__Q_fin = np.diag(np.concatenate((Q_fin*np.ones(N_p), np.zeros((3*N_p)))))
        if FinCost: self.__x_sp = np.hstack((x_sp, np.zeros(3*N_p)))
        if not FinCost: self.__x_sp = np.zeros(4*N_p)
        self.__bg = scc
        self.__first_iter = "true"
        self.__F_int = gp_dynamics.MDS_system() # Integrator used for next value in NLP
        self.__gp_dynamics = gp_dynamics
        self.__N = interval  # number of control intervals within MPC horizon
        self.__T = horizon  # time horizon of MPC window in seconds
        self.__dt = horizon/interval
        if Opti_MBK == False:
            self.__N_x = 4*N_p  # number of states of ode
            self.__N_u = N_p  # number of inputs of ode
        else:
            self.__N_x = 7 * N_p
            self.__N_u = 4 * N_p
        self.__N_p = N_p
        self.__x_init = list(np.zeros(self.__N_x)) #starting point (override in every MPC iteration)
        self.__x0 = None # initial guess for X
        self.__lam_x0 = None  # initial guess for lagrange multipliers for bounds on X
        self.__lam_g0 = None  # initial guess for lagrange multipliers for bounds on G
        # l:lower / u:upper bound on x:control/u:input
        if lbx == None: lbx = list(-np.ones(N_x)*np.inf)
        if ubx == None: ubx = list(np.ones(N_x)*np.inf)
        if lbu == None: lbu = list(-np.ones(N_u))
        if ubu == None: ubu = list(np.ones(N_u))
        self.__lbx = lbx #lower boundary on statespace
        self.__ubx = ubx #upper boundary on statespace
        self.__lbu = lbu #lower boundary on input
        self.__ubu = ubu #upper boundary on input
        

        self.precomp = precomp
        # solver options
        # Performance Change:
        # 'ipopt.mu_strategy' : 'monotone' instead of 'adaptive' ?
        #
        if Save_IPOPT == False: Out_file_Name = None
        if print_IPOPT:
            self.pipe_path = os.getcwd()+"/logs/mpc_log"
            if os.path.exists(self.pipe_path):
                os.remove(self.pipe_path)
            with open(self.pipe_path, "w") as p:
                p.write("***This is the IPOPT DEBUG WINDOW***\n   ~~~Prepare for amazing detail~~~ \n")
            print("Opened log on {}".format(self.pipe_path))

            self.out_file_path = os.getcwd()+"/logs/ipopt_print"
            Out_file_Name = self.out_file_path
        self.setIPOPT_opt(Out_file_Name, mu_strategy, bound_relax_factor, expect_infeasible_problem)

    def setIPOPT_opt(self, Out_file_Name, mu_strategy, bound_relax_factor, expect_infeasible_problem):
        self.options = {  # copied from the GP_MPC library @Kevin 9.4 @Christian: added some options
            'ipopt.linear_solver': 'MUMPS',           # Linear solver used for NLP
            # Output Options
            'ipopt.print_user_options': 'no',         # Print all options set by the user default 'no'
            'ipopt.print_options_documentation': 'no',# if selected, the algorithm will print the list of all available algorithmic options with some documentation before solving the optimization problem. The default value for this string option is "no"
            'ipopt.print_frequency_iter': 1,          # Determines at which iteration frequency the summarizing iteration output line should be printed. Default: 1
            'ipopt.output_file': Out_file_Name,       # File name of desired output file (leave unset for no file output).
            'ipopt.print_level' : 3,                  # Sets the default verbosity level for console output. The larger this value the more detailed is the output. The valid range for this integer option is 0 ≤ print_level ≤ 12 and its default value is 5.
            'ipopt.file_print_level': 5,              # Verbosity leve for output file. 0 - 12; Default is 5
            'ipopt.print_timing_statistics' : 'yes',  # If selected, the program will print the CPU usage (user time) for selected tasks. The default value for this string option is "no".
       
            # Termination (@Christian: so far all in default)
            'ipopt.tol' : 1e-5, #@Kevin 17.06                     # Desired convergence tolerance(relative). (Default is 1e-8)
            'ipopt.max_iter': 3000,                 # Maximum number of iterations (default is 3000)
            'ipopt.max_cpu_time' : 1e6,             # Maximum number of CPU seconds. (Default is 1e6)
            # Barrier Parameter
            'ipopt.mu_strategy': 'adaptive',        # Determines which barrier parameter update strategy is to be used. Default "monotone", can choose 'adaptive' instead
            'ipopt.mu_init' : 0.01,                 # This option determines the initial value for the barrier parameter (mu). It is only relevant in the monotone, Fiacco-McCormick version of the algorithm.
            'ipopt.mu_oracle' : mu_strategy,          # Oracle for a new barrier parameter (ONLY in the adaptive mu_strategy): Default is:  'quality-function' (:= minimize a quality function); alternatives are: 'probing' (:= Mehrotra's probing heuristic); 'loqo' (:= LOQO's centrality rule)
            # NLP
            'ipopt.check_derivatives_for_naninf': 'yes',
            'ipopt.bound_relax_factor': bound_relax_factor,                # Factor for initial relaxation of the bounds. Default: 1e-8
            # Warm Start
            'ipopt.warm_start_init_point' : 'yes',
            'ipopt.warm_start_bound_push' : 1e-9,           # default is 0.001
            'ipopt.warm_start_bound_frac' : 1e-9,           # default is 0.001
            'ipopt.warm_start_slack_bound_frac' : 1e-9,     # default is 0.001
            'ipopt.warm_start_slack_bound_push' : 1e-9,     # default is 0.001
            'ipopt.warm_start_mult_bound_push' : 1e-9,      # default is 0.001
            # Restoration Phase
            'ipopt.expect_infeasible_problem': expect_infeasible_problem,            # Enable heuristics to quickly detect an infeasible problem. Default 'no'
            'ipopt.expect_infeasible_problem_ctol': 0.0001,     # Threshold for disabling "expect_infeasible_problem" option.
            'ipopt.expect_infeasible_problem_ytol': 1e8,        # Multiplier threshold for activating "expect_infeasible_problem" option.
            'ipopt.start_with_resto': 'no',                     # Tells algorithm to switch to restoration phase in first iteration.
            'ipopt.soft_resto_pderror_reduction_factor': 0.9999,# Required reduction in primal - dual error in the soft restoration phase.
            'ipopt.required_infeasibility_reduction': 0.9,      # Required reduction of infeasibility before leaving restoration phase.
            'ipopt.bound_mult_reset_threshold': 1000,           # Threshold for resetting bound multipliers after the restoration phase.
            'ipopt.constr_mult_reset_threshold': 0,             # Threshold for resetting equality and inequality multipliers after restoration phase.
            'ipopt.evaluate_orig_obj_at_resto_trial': 'yes',    # Determines if the original objective function should be evaluated at restoration phase trial points.


            'ipopt.acceptable_constr_viol_tol' : 1e-10,
            'ipopt.constr_viol_tol' : 1e-8,
            #'ipopt.fixed_variable_treatment' : 'make_constraint',

            'print_time' : False,
            'verbose' : False,
            #'expand' : True
        }

    def solve(self, init_pose = None):
        # Set initial conditions:
        if init_pose.any() == None:
            print('No init pose given to MPC solve')
            init_pose = np.zeros(6)
        x_0 = np.zeros(self.__N_x)
        #Create input-dictionary for solver
        if self.__first_iter == "true":
            J, w, g, lbw, ubw, lbg, ubg, x_opt, u_opt, w0 = self.solverhelper(init_pose=init_pose)
            # Create an NLP solver
            prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
            if not self.precomp:
                self.solver = ca.nlpsol('solver', 'ipopt', prob, self.options) #without precompiling
            else:
                gen_opts={}
                #gen_opts['casadi_real'] = 'float'
                gen_opts['casadi_int'] = 'long int'
                solver = ca.nlpsol('solver', 'ipopt', prob, self.options)
                solver.generate_dependencies("nlp.c", gen_opts)
                start = time.time()
                subprocess.Popen("gcc -fPIC -shared -O1 nlp.c -o nlp.so", shell=True).wait()
                print("Compile time was: {}".format(time.time()-start))
                self.solver = ca.nlpsol("solver", "ipopt", "./nlp.so")

            # Function to get x and u trajectory from w
            self.extract_traj = ca.Function('opti_plot', [ca.vertcat(*w)], [ca.horzcat(*x_opt), ca.horzcat(*u_opt)], ['w'], ['x', 'u'])
            self.args = dict(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
            self.__first_iter = "false" # first iteration done
        else:
            self.args["lbx"][0:self.__N_x] = list(x_0)
            self.args["ubx"][0:self.__N_x] = list(x_0)
            self.__lbw[0:self.__N_x] = list(x_0)
            self.__ubw[0:self.__N_x] = list(x_0)
            self.__x0[0:self.__N_x] = list(x_0)
            args = dict(x0=self.__x0, lbx=self.__lbw, ubx=self.__ubw, lam_x0 = self.__lam_x0, lam_g0 = self.__lam_g0)

        # Adding init_pose to dynamics requires problem built new each time
        J, w, g, lbw, ubw, lbg, ubg, x_opt, u_opt, w0 = self.solverhelper(init_pose=init_pose)
        prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)} 
        self.solver = ca.nlpsol('solver', 'ipopt', prob, self.options) #without precompiling
        # Solve the NLP
        sol = self.solver(**self.args)
        self.__x0 = sol['x']
        self.__lam_x0 = sol['lam_x']
        self.__lam_g0 = sol['lam_g']
    
        x_plot, u_plot = self.extract_traj(sol['x'])

        X_array = x_plot.full()
        self.__x_plot = X_array
        self.x_traj = X_array # @Kevin 17.6.21: w/o __ is visible outside the class
        self.x_1 = X_array[:,1]
        self.__u_plot = u_plot.full()
        self.__x_init = x_plot.full()[:,1]

        if hasattr(self, "pipe_path"):
            try:
                with open(self.pipe_path, "w") as p, open(self.out_file_path) as ip:
                    for line in ip:
                        p.write(line)
                    p.write("\ninitial state:  \n{}\n".format(init_pose))
                    p.write("ctrl soln:      \n{}\n".format(u_plot.full()))
                    p.write("state soln:     \n{}\n".format(x_plot.full()))
                    p.write("cost breakdown: \n{}\n\n\n".format(self.__gp_dynamics.split_cost_function(X_array, self.__u_plot)))
            except BrokenPipeError:
                print(BrokenPipeError)
                pass

        return u_plot.full(), x_plot.full()

    def solverhelper(self, init_pose=None): # Formulate the NLP for multiple-shooting
        N = self.__N      # support-points of optimization trajectorie
        F_int = self.__F_int # integrator
        N_x = self.__N_x  # Number of States
        N_u = self.__N_u  # Number of Inputs

        if init_pose.any() == None:
            x_0 = init_pose[:,N_x]
        else:
            x_0 = np.zeros(N_x) # Initial state is now relative to start: always 0

        # Limits of states and inputs
        lbx = self.__lbx
        ubx = self.__ubx
        lbu = self.__lbu
        ubu = self.__ubu
        bg = self.__bg

        x_sp = self.__x_sp
        Q_fin = self.__Q_fin
        FinCost = self.__FinCost
        

        if self.__first_iter == "true" or self.__first_iter != "true":

            # Initialize empty NLP
            w = []    # Decision variables at the optimal solution (N_xc x 1)
            w0 = []   # Decision variables, initial guess (nxc x 1)
            lbw = []  # Decision variables lower bound (nxc x 1), default -inf.
            ubw = []  # Decision variables upper bound (nxc x 1), default +inf.
            J = 0     # Cost function value at the optimal solution (1 x 1)
            g = []    # Constraints function at the optimal solution (ng x 1)
            lbg = []  # Constraints lower bound (ng x 1), default -inf.
            ubg = []  # Constraints upper bound (ng x 1), default +inf.

            # For plotting x, u and force given w
            x_opt = []
            u_opt = []

            # "Lift" initial conditions
            X = x_0

            x_opt += [X]

            for k in range(N):
                # New NLP variable for the control
                Uk = ca.MX.sym('U_' + str(k), N_u)
                w += [Uk]
                lbw += lbu
                ubw += ubu
                w0 += list(np.zeros(N_u))
                u_opt += [Uk]

                # Integrate till the end of the interval
                Fk = F_int(x = X, u = Uk, init_pose = init_pose)
                X = Fk['xf']
                x_opt += [X]
                J_next = Fk['st_cost']
                J = J + J_next

            if FinCost == True:
                J = J + (Xk-x_sp).T@Q_fin@(Xk-x_sp)

            self.__J = J
            self.__w = w
            self.__g = g
            self.__lbw = lbw
            self.__ubw = ubw
            self.__lbg = lbg
            self.__ubg = ubg
            self.__x_opt = x_opt
            self.__u_opt = u_opt

        else:
            J = self.__J
            w = self.__w
            g = self.__g
            lbw = self.__lbw
            ubw = self.__ubw
            lbg = self.__lbg
            ubg = self.__ubg
            x_opt = self.__x_opt
            u_opt = self.__u_opt
            w0 = 0

        return J, w, g, lbw, ubw, lbg, ubg, x_opt, u_opt, w0
        


    def plot(self, plot=True, saveplot=False, Plt_file_Name = 'MPC_trajectories'):
        T = self.__T
        N = self.__N
        N_x = self.__N_x
        N_p = self.__N_p
        nx2 = int(N_x/2)
        nx4 = int(N_x/4)

        # Plot the solution
        x_plot =   self.__x_plot
        u_plot =   self.__u_plot
        tgrid = [T / N * k for k in range(N + 1)]

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        colors = 'rgb'
        plt.subplot(2,1,1)
        for ix in range(int(N_x/4)):
            c = colors[ix]
            plt.plot(tgrid, x_plot[ix,:], '--', color = c, label = 'x'+str(1+ix))
            plt.fill_between(tgrid, x_plot[ix]+x_plot[ix+nx2], x_plot[ix]-x_plot[ix+nx2], color = c, alpha = 0.5)
            plt.plot(tgrid, x_plot[ix+nx4,:], ':', color = c, label = 'x_dot'+str(1+ix))
            plt.fill_between(tgrid, x_plot[ix+nx4]+x_plot[ix+nx2+nx4], x_plot[ix+nx4]-x_plot[ix+nx2+nx4], color = c, alpha = 0.3)
        plt.legend()
        plt.grid()

        plt.subplot(2,1,2)
        for iu in range(N_p):
            plt.step(tgrid, np.append(np.nan, u_plot[iu,:]), '.-', color = colors[iu], label = 'u'+str(1+iu))
        plt.xlabel('t')
        plt.legend()
        plt.grid()
        if saveplot==True: plt.savefig(Plt_file_Name)
        if plot == True: plt.show()

        print("Optimal Delta M {} is".format(u_plot[1 * N_p: 2 * N_p, -1]))
        print("Optimal Delta B {} is".format(u_plot[2 * N_p: 3 * N_p, -1]))
        print("Optimal Delta K {} is".format(u_plot[3 * N_p:, -1]))

#____________________________________________________________________________________________________________________________________________


def euler_to_rotation(eu):
    rot = ca.MX.zeros(3,3)
    rot[0,0] =  ca.cos(eu[3])*ca.cos(eu[4])*ca.cos(eu[5]) - ca.sin(eu[3])*ca.sin(eu[5])
    rot[0,1] = -ca.cos(eu[3])*ca.cos(eu[4])*ca.sin(eu[5]) - ca.sin(eu[3])*ca.cos(eu[5])
    rot[0,2] =  ca.cos(eu[3])*ca.sin(eu[4])
    rot[1,0] =  ca.sin(eu[3])*ca.cos(eu[4])*ca.cos(eu[5]) + ca.cos(eu[3])*ca.sin(eu[5])
    rot[1,1] = -ca.sin(eu[3])*ca.cos(eu[4])*ca.sin(eu[5]) + ca.cos(eu[3])*ca.cos(eu[5])
    rot[1,2] =  ca.sin(eu[3])*ca.sin(eu[4])
    rot[2,0] = -ca.sin(eu[4])*ca.cos(eu[5])
    rot[2,1] =  ca.sin(eu[4])*ca.sin(eu[5])
    rot[2,2] =  ca.cos(eu[4])
    return rot

class GPDynamics:

    def __init__(self, N_p, Opti_MBK = False, gp_extern = None, dt=7.5/20,
                 Q_pos=0.0, Q_vel=0.05, R=0.05, S=0.015, H=0.2, I=0.3,
                 M_k = 2.0, B_k = 10.0, K_k = 0, x_0 = None, x_sp = None ):

        # Parameters
        self.__Opti_MBK = Opti_MBK
        self.__x_sp = x_sp # Point where the human wants to go  (ONLY used in artificial data for human GP Model)
        self.__x_0 = x_0   # start point
        self.__N_p = N_p   # Number of Positions                (3 or 6 depending on with or without rotation in 3D space)
        self.__N_u = N_p   # Number of Inputs for the system    (N_u = N_p + 3 Np -> force + 3 * diagonal elements of MBK matrices of system dynamic)
        self.__N_x = 2 * N_p   # Number of states of system         (positions & d/dt positions)
        self.__N_xc = 4 * N_p   # Number of states in full state space
        self.__dt = dt
        self.__Q_pos = Q_pos    # stagecost on positions
        self.__Q_vel = Q_vel    # stagecost on velocities
        self.__R = R            # stagecost on input
        self.__S = S            # stagecost on covariance
        self.__H = H            # stagecost on human forces
        self.__I = I
        self.__M_k = M_k        # M Matrix Diagonal entries (without delta part)
        self.__B_k = B_k        # B Matrix Diagonal entries (without delta part)
        self.__K_k = K_k        # K Matrix Diagonal entries (without delta part)
        if gp_extern:
            self.__gp = gp_extern
        else:
            self.__gp = None

    def GP_init(self, GP_size = 'medium', data = None, split_ratio = 0.8): # Build a GP for human forces from experimental or artificial data
    #Parameter
    # data:             dictionary with two entries:
    #                                       X is positions (3D)
    #                                       Y is forces (3D/6D with rotation) at position X
    # split_ratio:      ratio for test to training data
    #                                       GP is generated with N_data * split ratio
    #                                       GP is testet with  N_data * (1 - split ratio)
        if self.__gp:
            print("GP already initialized, but loading new data")
        if data == None:
            if GP_size == 'small':
                spaceing = 3.33333333
            if GP_size == 'medium':
                spaceing = 2.5
            if GP_size == 'large':
                spaceing = 2


            f_max = 100  # Maximum force applied by human
            noise = 0.2  # Scale of additive noise to human forces
            x_sp = self.__x_sp
            #Meshgrid of points in space
            x, y, z = np.meshgrid(np.arange(-5, 5.1, spaceing),
                                  np.arange(-5, 5.1, spaceing),
                                  np.arange(-5, 5.1, spaceing), sparse=False, indexing='ij')

            #forces are poroportional to distance from x_sp (within limits)
            u = x_sp[0] - x + noise*np.random.randn(x.shape[0], x.shape[1], x.shape[2])
            v = x_sp[1] - y + noise*np.random.randn(x.shape[0], x.shape[1], x.shape[2])
            w = x_sp[2] - z + noise*np.random.randn(x.shape[0], x.shape[1], x.shape[2])

            #Limit applied forces to f_max
            u[u > f_max] = f_max
            u[u < -f_max] = -f_max
            v[v > f_max] = f_max
            v[v < -f_max] = -f_max
            w[w > f_max] = f_max
            w[w < -f_max] = -f_max

            #flatten and stack for input to gp
            X_pos=np.vstack((x.flatten(),y.flatten(),z.flatten()))
            Y_force=np.vstack((u.flatten(),v.flatten(),w.flatten()))

        else:
            X_pos = data['X']
            Y_force = data['Y']

        #Split with ratio 0.8 -> 80% training / 20% test
        x_train, y_train, x_test, y_test = split(X_pos, Y_force, split_ratio)

        #gp.predict wants (Dim,N_samples) not (N_samples,Dim)
        x_train = x_train.T
        y_train = y_train.T
        x_test = x_test.T
        y_test = y_test.T

        # Create GP model and optimize hyper-parameters
        gp = GP(x_train, y_train, mean_func='zero', normalize=False, optimizer_opts=None, optimize_nummeric=False)
        gp.validate(x_test, y_test)

        self.__gp = gp


    def MDS_system(self,M = None, B = None, K = None):  # Build discrete dynamics for later integration

        #Parameters
        N_p = self.__N_p    # Num of positions in system
        N_x = self.__N_x    # X is position and velocity
        N_xc = self.__N_xc  # Full state space
        N_u = self.__N_u    # U is input if u = [fd, DeltaMBKdiagonal]
        dt = self.__dt      # time step for discretization
        Q_pos = self.__Q_pos    # stagecost on positions
        Q_vel = self.__Q_vel    # stagecost on velocities
        R = self.__R            # stagecost on input
        S = self.__S            # stagecost on covariance
        H = self.__H            # stagecost on human forces
        Ic = self.__I            # stagecost on human force covariance
        if M ==None: M_k = self.__M_k        # M Matrix Diagonal entries (without delta part)
        else: M_k = M
        if B == None: B_k = self.__B_k        # B Matrix Diagonal entries (without delta part)
        else: B_k = B
        if K == None: K_k = self.__K_k        # K Matrix Diagonal entries (without delta part)
        else: K_k = K
        gp = self.__gp          # gaussian process
        Opti_MBK = self.__Opti_MBK #Optimize MBK (System) Matrices
        
        # Rewrite, @Kevin 11.6.21, try to reduce matrices, only use N_p
        u = ca.MX.sym('u', N_p+Opti_MBK*3*N_p)
        x = ca.MX.sym('x', 4*N_p+Opti_MBK*3*N_p)
        x_pos_cov = ca.diag(x[2*N_p:3*N_p])

        init_pose = ca.MX.sym('init_pose', 6)
        T_c_w_0 = euler_to_rotation(init_pose)
        x_w = T_c_w_0 @ x[0:3]+init_pose[0:3] # x is in compliance frame, rotate+add initial position
        #f_mu, f_cov = gp.predict(x=x[:N_p], u = [], cov=x_pos_cov) # Old calc, x in compliance frame
        f_mu, f_cov = gp.predict(x=x_w, u = [], cov=x_pos_cov)
        f_jac = gp.jacobian(x_w, [], x_pos_cov)
        x_next = ca.MX(4*N_p+Opti_MBK*3*N_p,1)
        for i in range(N_p):
            if Opti_MBK:
                x_next[i+4*N_p] = x[i+4*N_p]+u[1*N_p+i]
                x_next[i+5*N_p] = x[i+5*N_p]+u[2*N_p+i]
                x_next[i+6*N_p] = x[i+6*N_p]+u[3*N_p+i]
                M_k = x[i+4*N_p]
                B_k = x[i+5*N_p]
                K_K = x[i+6*N_p]

            kn = dt*K_k/M_k
            bn = 1.0-dt*B_k/M_k
            btn = dt/M_k

            x_next[i] = x[i]+dt*x[i+N_p]+0.5*dt*dt*(-f_mu[i]+u[i])  # pos
            x_next[i+N_p] =  -kn*x[i]+bn*x[i+N_p]+btn*(-f_mu[i]+u[i]) # vel
            #x_next[i+N_p] =  ca.exp(-B_k/M_k*dt)*x[i+N_p]+btn*(-f_mu[i]+u[i]) # vel
            x_next[i+2*N_p] = x[i+2*N_p]+dt*dt*x[i+3*N_p] # cov pos
            x_next[i+3*N_p] = kn**2*x[i+2*N_p]+bn**2*x[i+3*N_p]+btn**2*f_cov[i,i]+btn**2*f_jac[i,i]**2*x[i+2*N_p] # cov vel

        L = Q_pos*ca.sumsqr(x[:N_p]) + Q_vel*ca.sumsqr(x[N_p:2*N_p]) + R*ca.sumsqr(u[:N_p])+ S*ca.sum1(x[2*N_p:]) + H*ca.sumsqr(f_mu) + Ic*ca.trace(f_cov) # Define Costfunction L (only stagecost)
        if Opti_MBK:
            L += 0.01*ca.sumsqr(u[N_p:])
        dynamics = ca.Function('F_int', [x, u, init_pose],[x_next, L], ['x', 'u', 'init_pose'], ['xf', 'st_cost'])
        return dynamics

    def split_cost_function(self, x_traj, u_traj = None):
    #returns a string of the contribution of various parts of the cost function
    
        types = ['pos', 'vel', 'x_cov', 'f', 'f_cov']
        if u_traj is not None: types += 'u'
        cost_total = {typ:0.0 for typ in types}
        Np = self.__N_p
        for x in x_traj:
            cost_total['pos'] += self.__Q_pos*x[:Np].T @ x[:Np]
            cost_total['vel'] += self.__Q_vel*x[Np:2*Np].T @ x[Np:2*Np]
            cost_total['x_cov'] += self.__S*x[2*Np:].T @ x[2*Np:]
            f_mu, f_cov = self.__gp.predict(x=x[:Np], u = [], cov=x[2*Np:3*Np])
            cost_total['f'] += self.__H*f_mu.T@f_mu
            cost_total['f_cov'] += self.__I*np.trace(f_cov)
        for u in u_traj:
            cost_total['u'] += self.__R*np.sum(u**2)
        return cost_total
#___________________________________________________________________________________________________________________________________________

#Helperfunctions (maybe assign to class?)

#Split Data in test and training data:
def split(x, y, ratio):
    # training datalength = ratio * datalength
    # test datalength = (1-ratio) * datalength
    if x.shape[1] != y.shape[1]:
        print("Can not split data: x amd y have size {} and {} but must be equal!".format(x.shape, y.shape))
        return
    if 0 >= ratio or 1 <= ratio:
        print("Can not split data: ratio is {} but must be between 0 and 1!".format(ratio))
        return
    N = x.shape[1]
    indices = np.arange(0, N, 1)
    np.random.shuffle(indices)
    x_train = x[:, np.sort(indices[:int(ratio*N)], axis=None)]
    x_test = x[:, np.sort(indices[int(ratio*N)+1:], axis=None)]
    y_train = y[:, np.sort(indices[:int(ratio*N)], axis=None)]
    y_test = y[:, np.sort(indices[int(ratio*N)+1:], axis=None)]
    return x_train, y_train, x_test, y_test

# Build constraints from limits
def constraints(N_p, pos_max = 100.0, vel_max = 10.0, cov_max = 1.0e3,
                u_x_max = 100, u_y_max = 100, u_z_max = 100,
                u_M_max = 40, u_B_max = 5000, u_K_max = 2000,
                u_dM_max = 1, u_dB_max = 10, u_dK_max = 5, Opti_MBK= False):
    # constraints for positions an velocities
    xlb = list(np.concatenate((np.full(N_p, -pos_max), np.full(N_p, -vel_max), np.full(2*N_p, 1e-6))))
    xub = list(np.concatenate((np.full(N_p, pos_max), np.full(N_p, vel_max), np.full(2*N_p, cov_max))))

    # constraints for input
    ulb = [-u_x_max, -u_y_max, -u_z_max]
    uub = [u_x_max, u_y_max, u_z_max]
    u_MBK_max = [u_M_max, u_B_max, u_K_max]
    u_dMBK_max = [u_dM_max, u_dB_max, u_dK_max]

    # if MBK are part of u
    if Opti_MBK == True:
        for i in range(3):  # for every matrice M B K
            for ii in range(N_p):  # for every entry on diagonal
                ulb.append(-u_dMBK_max[i])
                uub.append(u_dMBK_max[i])
                xlb.append(0.0)
                xub.append(u_MBK_max[i])

    return xlb, xub, ulb, uub

#____________________________________________________________________________________________________________________________________________


if __name__ == '__main__':
    # All important Parameters at one place
    np.random.seed(0)
    Opti_MBK = False

    # System Dynamics
    N_p = 3             # Num of positions in system
    N_xc = (4+3*Opti_MBK)*N_p  # Full state space
    x_0 = np.zeros(N_xc)   # starting point for system
    init_pos = np.zeros(6) 
    T = 7.5                # time horizon in seconds
    N = 50                # interval within time horizon
    dt=T/N

    # Gaussian Process
    x_sp = np.array([0.2, -0.1, 0.3,])

    #Cost weigths: if Q=R=S cost is equal for states input and variance in human forces
    Q_fin = 1000
    Q_pos = 0.0 # stagecost on positions
    Q_vel = 3.0 # stagecost on velocities
    R = 0.1     # stagecost on input
    S = 5.0     # stagecost on state covariance
    H = 5.0     # stagecost on human forces
    I = 0.1     # stagecost on human forces cov
    M_k = 5     # M Matrix Diagonal entries (without delta part)
    B_k = 10    # B Matrix Diagonal entries (without delta part)
    K_k = 0     # K Matrix Diagonal entries (without delta part)
    if Opti_MBK:
        x_0[4*N_p:5*N_p] = M_k
        x_0[5*N_p:6*N_p] = B_k
        x_0[6*N_p:7*N_p] = K_k

    # constraints for positions an velocities
    pos_max = 100.0
    vel_max = 10.0
    cov_max = 1.0e3

    # constraints for input
    u_x_max = 100
    u_y_max = 100
    u_z_max = 100
    # if Opti_MBK == True
    u_M_max = 50
    u_B_max = 2000
    u_K_max = 5000

    Save_IPOPT = False      # Save IPOpt Report
    FinCost = False          # Use final cost

    # IPOPT parameter
    scc = 0                                     # slack for continious constraints violation
    bound_relax_factor = 1e-7                   # 1e-7, 1e-8 # slack for all constraints violation
    mu_strategy = 'monotone'                    # ['monotone', 'adaptive']
    expect_infeasible_problem = 'yes'           # ['yes', 'no']



    # MPC linear solver for NLP
    solver = 'Mumps'

        
    #Get boundaries as constraints for optimal control problem
    xlb, xub, ulb, uub = constraints(N_p, pos_max, vel_max, cov_max, u_x_max, u_y_max, u_z_max, u_M_max, u_B_max, u_K_max, Opti_MBK=Opti_MBK)

    #Init Dynamic model of system including
    GP_Dynamics = GPDynamics(N_p=N_p, dt=dt, Q_pos=Q_pos, Q_vel=Q_vel, R=R, S=S, H=H, I=I, M_k=M_k, B_k=B_k, K_k=K_k, x_0=x_0, x_sp=x_sp, Opti_MBK=Opti_MBK)

    # Save Time stats
    f = open("Time_stats", "w")

    # Build permutations of OPOPT settings
    bound_relax_factor_list = [1e-5, 1e-6, 1e-7]  # all constraints violation
    mu_strategy_list = ['probing', 'quality-function', 'loqo']
    expect_infeasible_problem_list = ['yes', 'no']
    #GP_size_list = ['small', 'medium', 'large']
    GP_size_list = ['small']
    all_var = [bound_relax_factor_list, mu_strategy_list, expect_infeasible_problem_list, GP_size_list]
    all_permutations = list(itertools.product(*all_var))


    Global_Iter_range = 1
    iter_range = 2
    #Build permutations of IPOPT settings
    for permu in all_permutations[0:1]:

        #IPOPT settings as permutation of different options
        bound_relax_factor = permu[0]
        mu_strategy = permu[1]
        expect_infeasible_problem = permu[2]
        GP_size = permu[3]

        #Save time of calculation outside of loop
        time_solve_first_mean = 0
        time_solve_second_mean = 0

        # Global iteration necessary because time of calculation is not deterministic
        for Global_Iter in range(Global_Iter_range):

            #start with fresh x_0 (for new cold start)
            x_0 = np.zeros(N_xc)
            if Opti_MBK:
                x_0[4*N_p:5*N_p] = M_k
                x_0[5*N_p:6*N_p] = B_k
                x_0[6*N_p:7*N_p] = K_k
 
            
            # Init gaussian process for human forces in three dim space
            GP_Dynamics.GP_init(
                GP_size=GP_size)  # if there is real world data do: GP_Dynamics.GP_init(data=Dataset_dict, split_ratio=ratio)

            # Generate filename from IPOPT Options and iter-count
            IP_Opt_Result_file = "Results_for__Iteration_" + str(Global_Iter) + '_loop_' + str(
                iter) + "___" + 'bound_relax_factor_' + str(
                bound_relax_factor) + '___' + 'mu_strategy_' + str(
                mu_strategy) + '___' + 'GP_size_' + GP_size + '___' + 'expect_infeasible_problem_' + str(expect_infeasible_problem)

            # init MPC
            mpc = MPC(x_sp=x_sp, Q_fin=Q_fin, FinCost=FinCost, Save_IPOPT=Save_IPOPT, Opti_MBK=Opti_MBK, Out_file_Name=IP_Opt_Result_file, mu_strategy=mu_strategy,
                      expect_infeasible_problem=expect_infeasible_problem, bound_relax_factor=bound_relax_factor,
                      scc=scc,
                      gp_dynamics=GP_Dynamics, N_p=N_p, horizon=T, interval=N,
                      lbx=xlb, ubx=xub, lbu=ulb, ubu=uub, precomp = False)


            # one cold- and iter_range - 1 warm-starts
            for iter in range(iter_range):

                # Generate filename from IPOPT Options and iter-count
                IP_Opt_Result_file = "Results_for__Iteration_" + str(Global_Iter) + '_loop_' + str(
                    iter) + "___" + 'bound_relax_factor_' + str(
                    bound_relax_factor) + '___' + 'mu_strategy_' + str(
                    mu_strategy) + '___' + 'expect_infeasible_problem_' + str(expect_infeasible_problem)

                #Set IPOPT settings: mostly for new file name once per iterange for other parameter update
                mpc.setIPOPT_opt(Out_file_Name=IP_Opt_Result_file, mu_strategy=mu_strategy,
                                 bound_relax_factor=bound_relax_factor,
                                 expect_infeasible_problem=expect_infeasible_problem)


                time_0 = time.time()                                    # Get start time
                #init_pos[0] = -0.1 # Testing that the rotations work ;)
                init_pos[3] = 3.1415/2 # Testing that the rotations work ;)
                mpc.solve(init_pos)                                          # Solve the Optimal control problem
                time_solve = round((time.time() - time_0) * 1000)       # Get end time
                if iter == 0: time_solve_first_mean += time_solve       # save "cold start time"
                else: time_solve_second_mean += time_solve              # save "warm start time"
                x_0 = mpc.x_1                                        # save new x0 (x1 from last Iter) for next warmstart



            mpc.plot(saveplot=False, plot=True, Plt_file_Name=IP_Opt_Result_file)

        # Get mean time from sum
        time_solve_first_mean = time_solve_first_mean/Global_Iter_range
        time_solve_second_mean = time_solve_second_mean/((iter_range-1)*Global_Iter_range)

        # Save Time stats
        f.write(str(time_solve_first_mean))
        f.write(" Milliseconds for first iteration and ")
        f.write(str(time_solve_second_mean))
        f.write(IP_Opt_Result_file)
        f.write("\n")
    f.close()


