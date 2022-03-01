# MPC libary

# Includes
import numpy as np
import casadi as ca
from mpl_toolkits.mplot3d import axes3d #for Data generation for GP training
import matplotlib.pyplot as plt
from gp_mpc import Model, GP, MPC, plot_eig, lqr
import itertools
import time
from multiprocessing import Process, Queue
import matplotlib.animation as animation
#____________________________________________________________________________________________________________________________________________


class MPC:
    def __init__(self, N_p, horizon, interval, x_sp, Q_fin=1, FinCost= False, Save_IPOPT = False, Opti_MBK = False, mu_strategy = 'monotone', Out_file_Name = "IP_Opt_Results", expect_infeasible_problem='yes', bound_relax_factor=1e-8, scc = 0,  discrete_dynamics = None, lbx = None, ubx = None, lbu = None, ubu = None):

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
        self.__x_sp = np.hstack((x_sp, np.zeros(3*N_p)))
        self.__bg = scc
        self.__first_iter = "true"
        self.__F_int = discrete_dynamics # Integrator used for next value in NLP
        self.__N = interval  # number of control intervals within MPC horizon
        self.__T = horizon  # time horizon of MPC window in seconds
        self.__dt = horizon/interval
        self.__N_x = 4*N_p  # number of states of ode
        if Opti_MBK == False: self.__N_u = N_p  # number of inputs of ode
        else: self.__N_u = 4 * N_p
        self.__N_p = N_p
        self.__x_init = list(np.zeros(4*N_p)) #starting point (override in every MPC iteration)
        self.__x0 = None # initial guess for X
        self.__lam_x0 = None  # initial guess for lagrange multipliers for bounds on X
        self.__lam_g0 = None  # initial guess for lagrange multipliers for bounds on G
        # l:lower / u:upper bound on x:control/u:input
        if lbx == None: lbx = list(-np.ones(4*N_p)*np.inf)
        if ubx == None: ubx = list(np.ones(4*N_p)*np.inf)
        if lbu == None: lbu = list(-np.ones(N_p))
        if ubu == None: ubu = list(np.ones(N_p))
        self.__lbx = lbx #lower boundary on statespace
        self.__ubx = ubx #upper boundary on statespace
        self.__lbu = lbu #lower boundary on input
        self.__ubu = ubu #upper boundary on input
        # solver options
        # Performance Change:
        # 'ipopt.mu_strategy' : 'monotone' instead of 'adaptive' ?
        #
        if Save_IPOPT == False: Out_file_Name = None

        self.options = {  # copied from the GP_MPC library @Kevin 9.4 @Christian: added some options
            'ipopt.linear_solver': 'MUMPS',  # Linear solver used for NLP
            # Output Options
            'ipopt.print_user_options': 'yes',  # Print all options set by the user default 'no'
            'ipopt.print_options_documentation': 'no',
            # if selected, the algorithm will print the list of all available algorithmic options with some documentation before solving the optimization problem. The default value for this string option is "no"
            'ipopt.print_frequency_iter': 1,
            # Determines at which iteration frequency the summarizing iteration output line should be printed. Default: 1
            'ipopt.output_file': Out_file_Name,  # File name of desired output file (leave unset for no file output).
            'ipopt.print_level': 5,
            # Sets the default verbosity level for console output. The larger this value the more detailed is the output. The valid range for this integer option is 0 ≤ print_level ≤ 12 and its default value is 5.
            'ipopt.file_print_level': 5,  # Verbosity leve for output file. 0 - 12; Default is 5
            'ipopt.print_timing_statistics': 'yes',
            # If selected, the program will print the CPU usage (user time) for selected tasks. The default value for this string option is "no".
            # Termination (@Christian: so far all in default)
            'ipopt.tol': 1e-8,  # Desired convergence tolerance(relative). (Default is 1e-8)
            'ipopt.max_iter': 3000,  # Maximum number of iterations (default is 3000)
            'ipopt.max_cpu_time': 1e6,  # Maximum number of CPU seconds. (Default is 1e6)
            # Barrier Parameter
            'ipopt.mu_strategy': mu_strategy,
            # Determines which barrier parameter update strategy is to be used. Default "monotone", can choose 'adaptive' instead
            'ipopt.mu_init': 0.01,
            # This option determines the initial value for the barrier parameter (mu). It is only relevant in the monotone, Fiacco-McCormick version of the algorithm.
            'ipopt.mu_oracle': 'probing',
            # Oracle for a new barrier parameter (ONLY in the adaptive mu_strategy): Default is:  'quality-function' (:= minimize a quality function); alternatives are: 'probing' (:= Mehrotra's probing heuristic); 'loqo' (:= LOQO's centrality rule)
            # NLP
            'ipopt.check_derivatives_for_naninf': 'yes',
            'ipopt.bound_relax_factor': bound_relax_factor,
            # Factor for initial relaxation of the bounds. Default: 1e-8
            # Warm Start
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-9,  # default is 0.001
            'ipopt.warm_start_bound_frac': 1e-9,  # default is 0.001
            'ipopt.warm_start_slack_bound_frac': 1e-9,  # default is 0.001
            'ipopt.warm_start_slack_bound_push': 1e-9,  # default is 0.001
            'ipopt.warm_start_mult_bound_push': 1e-9,  # default is 0.001
            # Restoration Phase
            'ipopt.expect_infeasible_problem': expect_infeasible_problem,
            # Enable heuristics to quickly detect an infeasible problem. Default 'no'
            'ipopt.expect_infeasible_problem_ctol': 0.0001,
            # Threshold for disabling "expect_infeasible_problem" option.
            'ipopt.expect_infeasible_problem_ytol': 1e8,
            # Multiplier threshold for activating "expect_infeasible_problem" option.
            'ipopt.start_with_resto': 'no',  # Tells algorithm to switch to restoration phase in first iteration.
            'ipopt.soft_resto_pderror_reduction_factor': 0.9999,
            # Required reduction in primal - dual error in the soft restoration phase.
            'ipopt.required_infeasibility_reduction': 0.9,
            # Required reduction of infeasibility before leaving restoration phase.
            'ipopt.bound_mult_reset_threshold': 1000,
            # Threshold for resetting bound multipliers after the restoration phase.
            'ipopt.constr_mult_reset_threshold': 0,
            # Threshold for resetting equality and inequality multipliers after restoration phase.
            'ipopt.evaluate_orig_obj_at_resto_trial': 'yes',
            # Determines if the original objective function should be evaluated at restoration phase trial points.

            'ipopt.acceptable_constr_viol_tol': 1e-10,
            'ipopt.constr_viol_tol': 1e-8,
            'ipopt.fixed_variable_treatment': 'make_constraint',

            'print_time': False,
            'verbose': False,
            # 'expand' : True

        }


    def setIPOPT_opt(self,Out_file_Name,mu_strategy,bound_relax_factor,expect_infeasible_problem):
        self.options = {  # copied from the GP_MPC library @Kevin 9.4 @Christian: added some options
            'ipopt.linear_solver': 'MUMPS',           # Linear solver used for NLP
            # Output Options
            'ipopt.print_user_options': 'yes',       # Print all options set by the user default 'no'
            'ipopt.print_options_documentation': 'no', # if selected, the algorithm will print the list of all available algorithmic options with some documentation before solving the optimization problem. The default value for this string option is "no"
            'ipopt.print_frequency_iter': 1,        # Determines at which iteration frequency the summarizing iteration output line should be printed. Default: 1
            'ipopt.output_file': Out_file_Name,     # File name of desired output file (leave unset for no file output).
            'ipopt.print_level' : 5,                # Sets the default verbosity level for console output. The larger this value the more detailed is the output. The valid range for this integer option is 0 ≤ print_level ≤ 12 and its default value is 5.
            'ipopt.file_print_level': 5,           # Verbosity leve for output file. 0 - 12; Default is 5
            'ipopt.print_timing_statistics' : 'yes', # If selected, the program will print the CPU usage (user time) for selected tasks. The default value for this string option is "no".
            # Termination (@Christian: so far all in default)
            'ipopt.tol' : 1e-8,                     # Desired convergence tolerance(relative). (Default is 1e-8)
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
            'ipopt.fixed_variable_treatment' : 'make_constraint',

            'print_time' : False,
            'verbose' : False,
            # 'expand' : True

        }

    def solve(self, x_0 = None):

        # Set initial conditions:
        if x_0.any() == None:
            x_0 = self.__x_init

        # @Kevin: maybe we can move the dec variable, bounds, initial guess to a helper function? -> yes we can :)
        J, w, g, lbw, ubw, lbg, ubg, x_opt, u_opt, w0 = self.solverhelper(x_0=x_0)

        # Create an NLP solver
        prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
        solver = ca.nlpsol('solver', 'ipopt', prob, self.options)

        # Function to get x and u plot of optimal solution from w
        opti_plot = ca.Function('opti_plot', [ca.vertcat(*w)], [ca.horzcat(*x_opt), ca.horzcat(*u_opt)], ['w'], ['x', 'u'])

        #Create input-dictionary for solver
        if self.__first_iter == "true":
            args = dict(x0=w0,
                        lbx=lbw,
                        ubx=ubw,
                        lbg=lbg,
                        ubg=ubg)
        else:
            args = dict(x0=self.__x0,
                        lbx=lbw,
                        ubx=ubw,
                        lbg=lbg,
                        ubg=ubg,
                        lam_x0 = self.__lam_x0,
                        lam_g0 = self.__lam_g0)

        # Solve the NLP
        sol = solver(**args)
        self.__x0 = sol['x']
        self.__lam_x0 = sol['lam_x']
        self.__lam_g0 = sol['lam_g']
        # Rename? extract_traj?
        x_plot, u_plot = opti_plot(sol['x'])

        X_array = x_plot.full()
        self.__x_plot = X_array
        self.__x_1 = X_array[:,1]
        self.__u_plot = u_plot.full()
        self.__x_init = x_plot.full()[:,1]
        self.__first_iter = "false" # first iteration done
        return u_plot.full()

    def getX1(self):
        return self.__x_1

    def get_x_traj(self):
        return self.__x_plot

    def solverhelper(self, x_0): # Formulate the NLP for multiple-shooting

        N = self.__N      # support-points of optimization trajectorie
        F_int = self.__F_int # integrator
        N_x = self.__N_x  # Number of States
        N_u = self.__N_u  # Number of Inputs
        # Limits of states and inputs
        lbx = self.__lbx
        ubx = self.__ubx
        lbu = self.__lbu
        ubu = self.__ubu
        bg = self.__bg
        x_sp = self.__x_sp
        Q_fin = self.__Q_fin
        FinCost = self.__FinCost

        if self.__first_iter == "true":

            # Initialize empty NLP
            w = []  # Decision variables at the optimal solution (N_xc x 1)
            w0 = []  # Decision variables, initial guess (nxc x 1)
            lbw = []  # Decision variables lower bound (nxc x 1), default -inf.
            ubw = []  # Decision variables upper bound (nxc x 1), default +inf.
            J = 0  # Cost function value at the optimal solution (1 x 1)
            g = []  # Constraints function at the optimal solution (ng x 1)
            lbg = []  # Constraints lower bound (ng x 1), default -inf.
            ubg = []  # Constraints upper bound (ng x 1), default +inf.

            # For plotting x, u and force given w
            x_opt = []
            u_opt = []

            # "Lift" initial conditions
            Xk = ca.MX.sym('X_0', N_x)
            w += [Xk]
            lbw += list(x_0)
            ubw += list(x_0)
            w0 += list(x_0)
            bg0 = list(np.ones(N_x) * bg)

            # for plotting
            x_opt += [Xk]

            for k in range(N):
                # New NLP variable for the control
                Uk = ca.MX.sym('U_' + str(k), N_u)
                w += [Uk]
                lbw += lbu
                ubw += ubu
                w0 += list(np.zeros(N_u))
                u_opt += [Uk]

                # Integrate till the end of the interval
                Fk = F_int(x = Xk, u = Uk)
                Xk_next = Fk['xf']
                J_next = Fk['st_cost']
                J = J + J_next

                # New NLP variable for state at end of interval
                Xk = ca.MX.sym('X_' + str(k + 1), N_x)
                w += [Xk]
                lbw += list(lbx)
                ubw += list(ubx)
                w0 += list(np.zeros(N_x))

                x_opt += [Xk]

                # Add equality constraint
                g += [Xk_next - Xk]
                lbg += bg0
                ubg += bg0

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

            # "Lift" initial conditions
            lbw[0:N_x] = list(x_0)
            ubw[0:N_x] = list(x_0)


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

class GPDynamics:

    def __init__(self, N_p, Opti_MBK = False, gp_extern = None, dt=7.5/20, Q_pos=0.0, Q_vel=0.5, R=0.05, S=0.015, H=0.1, M_k = 30.0, B_k = 2000.0, K_k = 0, x_0 = None, x_sp = None ):

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
        gp = GP(x_train, y_train, mean_func='zero', normalize=True, optimizer_opts=None)#, optimize_nummeric=False)
        gp.validate(x_test, y_test)

        self.__gp = gp

    def MDS_system(self):  # Build discrete dynamics for later integration

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
        M_k = self.__M_k        # M Matrix Diagonal entries (without delta part)
        B_k = self.__B_k        # B Matrix Diagonal entries (without delta part)
        K_k = self.__K_k        # K Matrix Diagonal entries (without delta part)
        gp = self.__gp          # gaussian process
        Opti_MBK = self.__Opti_MBK #Optimize MBK (System) Matrices

        #Build Matrices for Cost function
        Q_stage = np.diag(np.concatenate((Q_pos*np.ones(N_p), Q_vel*np.ones(N_p)))) #stagecost on states
        R_stage = np.eye(N_p)*R #stagecost on input
        S_stage = np.eye(2 * N_p)*S #stagecost on covariance
        H_stage = H*np.eye(N_p)

        # Build Matrix symbolics of input / system states
        # for casadifunction of euler integration of systemdynamics (and costfkt)
        if Opti_MBK == False:
            u = ca.MX.sym('u',N_u)
        else:
            u = ca.MX.sym('u', (N_u+3*N_p))

        x = ca.MX.sym('x',N_xc)

        x_delta = x[:N_x]
        x_cov = ca.diag(x[N_x:])
        x_pos_cov = ca.diag(x[N_x:N_x+N_p])


        #Cobot Mass Damper system
        # Build MBK matrices (eye with konst + u)
        Delta_M = ca.MX.zeros(N_p, N_p)
        Delta_B = ca.MX.zeros(N_p, N_p)
        Delta_K = ca.MX.zeros(N_p, N_p)
        if Opti_MBK == True:
            for i in range(N_p): # u is on diagonal
                Delta_M[i, i] = M_k + u[1 * N_p + i]
                Delta_B[i, i] = B_k + u[2 * N_p + i]
                Delta_K[i, i] = K_k + u[3 * N_p + i]
        else:
            for i in range(N_p): # u is on diagonal
                Delta_M[i, i] = M_k
                Delta_B[i, i] = B_k
                Delta_K[i, i] = K_k

        M = Delta_M
        B = Delta_B
        K = Delta_K

        M_inv = ca.inv(M, 'symbolicqr')

        I = ca.MX.eye(N_p) #Identity Matrix
        O = ca.MX.zeros(N_p, N_p) #Zero Matrix
        A1 = ca.horzcat(I, I*dt)
        A2 = ca.horzcat(-dt*M_inv @ K, I-dt*M_inv @ B)
        At = ca.vertcat(A1, A2)
        Bt = ca.vertcat(O, M_inv*dt)

        f_mu, f_cov = gp.predict(x=x[:N_p], u = [], cov=x_pos_cov) # This way we reduce the num. of calls to GP, also, cov should be covariance in argument to GP
        x_next = At @ x[:N_x] + Bt @ u[:N_p] + Bt @ f_mu
        x_cov_next_full = At @ x_cov @ At.T + Bt @ f_cov @ Bt.T #TODO missing term w/ derivative of GP
        x_cov_next = []
        for i in range(N_x): x_cov_next = ca.vertcat(x_cov_next, x_cov_next_full[i,i])

        L = x_delta.T @ Q_stage @ x_delta + u[:N_p] .T @ R_stage @ u[:N_p] + ca.trace(x_cov_next_full @ S_stage) + f_mu.T @ H_stage @ f_mu # Define Costfunction L (only stagecost)

        dynamics = ca.Function('F_int', [x, u], [ca.vertcat(x_next, x_cov_next), L], ['x', 'u'], ['xf', 'st_cost'])

        return dynamics


#____________________________________________________________________________________________________________________________________________

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
                u_M_max = 0, u_B_max = 0, u_K_max = 0, Opti_MBK= False):
    # constraints for positions an velocities
    xlb = list(np.concatenate((np.full(N_p, -pos_max), np.full(N_p, -vel_max), np.full(2*N_p, 1e-6))))
    xub = list(np.concatenate((np.full(N_p, pos_max), np.full(N_p, vel_max), np.full(2*N_p, cov_max))))

    # constraints for input
    ulb = [-u_x_max, -u_y_max, -u_z_max]
    uub = [u_x_max, u_y_max, u_z_max]
    u_MBK_max = [u_M_max, u_B_max, u_K_max]

    # if MBK are part of u
    if Opti_MBK == True:
        for i in range(3):  # for every matrice M B K
            for ii in range(N_p):  # for every entry on diagonal
                ulb.append(-u_MBK_max[i])
                uub.append(u_MBK_max[i])

    return xlb, xub, ulb, uub

#____________________________________________________________________________________________________________________________________________

#Define MPC processes for later animation
def mpc_loop(x_0, communicator, iter = 10):
    for i in range(iter):
        mpc.solve(x_0)                                          # Solve the Optimal control problem
        x_0 = mpc.getX1()
        x_pos_traj = mpc.get_x_traj()
        communicator.put(x_pos_traj)
        #print("New x_0 = {}".format(x_0))
        #print("Iteration: {} finished".format(i))

#Plotupdate function
def update_lines(num, data, lines, communicator: Queue):
    global newdata
    print("Update Plot")
    if communicator.empty() == False:
        newdata = communicator.get()

        lines.set_data(newdata[0:2, :])
        lines.set_3d_properties(newdata[2, :])

    return lines


if __name__ == '__main__':
    # All important Parameters at one place

    # System Dynamics
    N_p = 3             # Num of positions in system: So far only 3 -> GP.Init() has to be changed such that it predicts Np values instead of 3
    x_0 = np.zeros(4*N_p) # starting point for system
    N_xc = 4*N_p        # Full state space
    T = 7.5             # time horizon in seconds
    N = 20              # interval within time horizon
    dt=T/N

    # Gaussian Process
    x_sp = np.array([2.2, 0.1, 0.3])

    #Cost weigths: if Q=R=S cost is equal for states input and variance in human forces
    Q_fin = 1000
    Q_pos = 0.0 # stagecost on positions
    Q_vel = 0.5 # stagecost on velocities
    R = 0.05    # stagecost on input
    S = 0.05    # stagecost on covariance
    H = 1     # stagecost on human forces
    M_k = 10  # M Matrix Diagonal entries (without delta part)
    B_k = 50  # B Matrix Diagonal entries (without delta part)
    K_k = 0     # K Matrix Diagonal entries (without delta part)

    # constraints for positions an velocities
    pos_max = 100.0
    vel_max = 10.0
    cov_max = 1.0e3

    # constraints for input
    u_x_max = 100
    u_y_max = 100
    u_z_max = 100
    # if Opti_MBK == True
    u_M_max = 0
    u_B_max = 0
    u_K_max = 0


    Opti_MBK = False         # Optimize MBK
    Save_IPOPT = False      # Save IPOpt Report
    FinCost = False          # Use final cost

    # IPOPT parameter
    scc = 0                                     # slack for continious constraints violation
    bound_relax_factor = 1e-5                   # 1e-7, 1e-8 # slack for all constraints violation
    mu_strategy = 'adaptive'                    #
    expect_infeasible_problem = 'no'           # ['yes', 'no']
    GP_size = 'small'


    # MPC linear solver for NLP
    solver = 'MUMPS'

    #Get boundaries as constraints for optimal control problem
    xlb, xub, ulb, uub = constraints(N_p, pos_max, vel_max, cov_max, u_x_max, u_y_max, u_z_max, u_M_max, u_B_max, u_K_max, Opti_MBK=Opti_MBK)

    # Init Dynamic model of system
    GP_Dynamics = GPDynamics(N_p=N_p, dt=dt, Q_pos=Q_pos, Q_vel=Q_vel, R=R, S=S, H=H, M_k=M_k, B_k=B_k, K_k=K_k,
                             x_0=x_0, x_sp=x_sp, Opti_MBK=Opti_MBK)

    # Init gaussian process for human forces in three dim space
    GP_Dynamics.GP_init(GP_size=GP_size)  # if there is real world data do: GP_Dynamics.GP_init(data=Dataset_dict, split_ratio=ratio)


    #start with fresh x_0 (for new cold start)
    x_0 = np.zeros(4 * N_p)

    # init MPC
    mpc = MPC(x_sp=x_sp, Q_fin=Q_fin, FinCost=FinCost, Save_IPOPT=Save_IPOPT, Opti_MBK=Opti_MBK, mu_strategy=mu_strategy,
              expect_infeasible_problem=expect_infeasible_problem, bound_relax_factor=bound_relax_factor,
              scc=scc,
              discrete_dynamics=GP_Dynamics.MDS_system(), N_p=N_p, horizon=T, interval=N,
              lbx=xlb,
              ubx=xub, lbu=ulb, ubu=uub)

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    dummydata = [[0] for index in range(10)]
    lineX = ax.plot(0, 0, 0)[0]

    # Setting the axes properties
    ax.set_xlim3d([-.1, 2.25])
    ax.set_xlabel('X')
    ax.set_ylim3d([-.1, .25])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-.1 , .4])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')



    communicator = Queue()

    print("start Process")
    duta = Process(target=mpc_loop, args=(x_0, communicator, 30))
    duta.start()

    line_ani = animation.FuncAnimation(fig, update_lines, 1, fargs=(dummydata, lineX, communicator), interval=500)
    plt.show()
    print("process finished")
    duta.join()
    print("Completed multiprocessing")








