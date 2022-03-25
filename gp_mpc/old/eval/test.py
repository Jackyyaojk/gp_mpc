
import itertools
import time

import numpy as np

from MPC_GP3 import MPC
from helper_fns import constraints, yaml_load, split
from gp_dynamics import GPDynamics
from gp_mpc import GP

def GP_init(GP_size = 'medium', data = None, split_ratio = 0.8, x_sp = None):
# Build a GP for human forces from experimental or artificial data
#Parameter
# data:             dictionary with two entries:
#                                       X is positions (3D)
#                                       Y is forces (3D/6D with rotation) at position X
# split_ratio:      ratio for test to training data
#                                       GP is generated with N_data * split ratio
#                                       GP is testet with  N_data * (1 - split ratio)

    if data == None:
        if GP_size == 'small':
            spaceing = 3.33333333
        if GP_size == 'medium':
            spaceing = 2.5
        if GP_size == 'large':
            spaceing = 2

        f_max = 100  # Maximum force applied by human
        noise = 0.2  # Scale of additive noise to human forces
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

    gp = GP(x_train, y_train, mean_func='zero', normalize=False, optimizer_opts=None, optimize_nummeric=False)
    gp.validate(x_test, y_test)
    return gp

#_____________________________________________________________________________________



# All important Parameters at one place
np.random.seed(0)
Opti_MBK = False

mpc_params = yaml_load('config/mpc_params.yaml')

# System Dynamics
N_p = 3             # Num of positions in system
N_xc = (4+3*Opti_MBK)*N_p  # Full state space
init_pos = np.zeros(6) 
T = 7.5                      # time horizon in seconds
mpc_params['mpc_pts'] = 20   # interval within time horizon
mpc_params['dt']=T/mpc_params['mpc_pts']


# Gaussian Process
x_sp = np.array([0.2, -0.1, 0.3,])

#Cost weigths: if Q=R=S cost is equal for states input and variance in human forces
mpc_params['Q_pos'] = 0.0 # stagecost on positions
mpc_params['Q_vel'] = 3.0 # stagecost on velocities
mpc_params['R'] = 0.1     # stagecost on input
mpc_params['S'] = 5.0     # stagecost on state covariance
mpc_params['H'] = 5.0     # stagecost on human forces
mpc_params['I'] = 0.1     # stagecost on human forces cov
M_k = 10      # M Matrix Diagonal entries (without delta part)
B_k = 1100    # B Matrix Diagonal entries (without delta part)
K_k = 0       # K Matrix Diagonal entries (without delta part)

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

# Save Time stats
f = open("logs/Time_stats", "w")

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
        GP = GP_init(GP_size=GP_size, x_sp = x_sp)  # if there is real world data do: GP_Dynamics.GP_init(data=Dataset_dict, split_ratio=ratio)
        GP_Dynamics = dict(single_mode = GPDynamics(N_p, mpc_params, GP))

        # Generate filename from IPOPT Options and iter-count
        IP_Opt_Result_file = "logs/Results_for__Iteration_" + str(Global_Iter) + '_loop_' + str(
            iter) + "___" + 'bound_relax_factor_' + str(
            bound_relax_factor) + '___' + 'mu_strategy_' + str(
            mu_strategy) + '___' + 'GP_size_' + GP_size + '___' + 'expect_infeasible_problem_' + str(expect_infeasible_problem)
        Out_file_Name = IP_Opt_Result_file
        # init MPC
        mpc = MPC(mpc_params=mpc_params, gp_dynamics_dict=GP_Dynamics, N_p=N_p)

        mpc.setIPOPT_opt(Out_file_Name, mu_strategy, bound_relax_factor, expect_infeasible_problem)
        # one cold- and iter_range - 1 warm-starts
        for iter in range(iter_range):

            # Generate filename from IPOPT Options and iter-count
            IP_Opt_Result_file = "logs/Results_for__Iteration_" + str(Global_Iter) + '_loop_' + str(
                iter) + "___" + 'bound_relax_factor_' + str(
                bound_relax_factor) + '___' + 'mu_strategy_' + str(
                mu_strategy) + '___' + 'expect_infeasible_problem_' + str(expect_infeasible_problem)

            #Set IPOPT settings: mostly for new file name once per iterange for other parameter update
            mpc.setIPOPT_opt(Out_file_Name=IP_Opt_Result_file, mu_strategy=mu_strategy,
                             bound_relax_factor=bound_relax_factor,
                             expect_infeasible_problem=expect_infeasible_problem)

            time_0 = time.time()                                    # Get start time
            mpc.solve(init_pos)                                     # Solve the Optimal control problem
            time_solve = round((time.time() - time_0) * 1000)       # Get end time
            if iter == 0: time_solve_first_mean += time_solve       # save "cold start time"
            else: time_solve_second_mean += time_solve              # save "warm start time"
            #x_0 = mpc.x_1                                           # save new x0 (x1 from last Iter) for next warmstart

        mpc.plot(saveplot=False, plot=True, Plt_file_Name=IP_Opt_Result_file)

    # Get mean time from sum
    time_solve_first_mean = time_solve_first_mean/Global_Iter_range
    time_solve_second_mean = time_solve_second_mean/((iter_range-1)*Global_Iter_range)
    print("First solve {}, warm start {}".format(time_solve_first_mean, time_solve_second_mean))
    # Save Time stats
    f.write(str(time_solve_first_mean))
    f.write(" Milliseconds for first iteration and ")
    f.write(str(time_solve_second_mean))
    f.write(IP_Opt_Result_file)
    f.write("\n")
f.close()


# Create GP model and optimize hyper-parameters
#gp = GP(x_train, y_train, mean_func='zero', normalize=False, optimizer_opts=None, optimize_nummeric=False)
#gp.validate(x_test, y_test)

