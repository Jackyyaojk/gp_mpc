import yaml
import numpy as np
from os import getcwd

# YAML functions for saving and loading dictionarys

def yaml_save(path, dictionary):
    with open(path, 'w') as file:
        documents = yaml.dump(dictionary, file, default_flow_style=False)

def yaml_load(path):
    yaml_file = open(path, 'r')
    # yaml_content = yaml.load(yaml_file, Loader=yaml.FullLoader) #YAML 5.1 doesnt like numpy arrays :(
    yaml_content = yaml.load(yaml_file, Loader=yaml.UnsafeLoader)
    local_list = []
    for key, value in yaml_content.items():
        local_list.append((key, value))
    return dict(local_list)

obs_range = 3
state_range = 3

mode_detector_params = {
    'bel_floor'     : -50.0,     # minimum log lik for mode
    'lik_floor'     : -20.0,     # minimum log lik for dimension in mode
    'bel_smoothing' : 3.0,       # weight on bt when updating b_{t+1}
    'anomaly_lik'   : -60.0,     # log lik of anomaly
    'print_belief'  : False,     # print mode belief
    'print_dof_lik' : False,     # print likelihood of each DOF
    'min_force'     : 3e0,       # below this ||f|| no belief updates
}

mpc_params = {
    'dt'     : 0.1,    # Horizon in sec
    'mpc_pts': 10,     # Number of MPC points
    'Q_pos'  : 0.0,    # Position
    'Q_vel'  : 0.1,    # Velocity
    'S'      : 0.2,    # Covariance
    'R'      : 5e0,    # Control
    'H'      : 1e0,    # Human forces
    'I'      : 1.75e0, # Human covariance
    'cov_max': 1e10,   # max covariance
    'print_IPOPT': False,  # IPOPT output
    'precomp'    : False,  # Precompile the solver
    'live_plot'  : False,  # Whether to use live-plot
    'human_kin'  : {'center': [0.5, 0.0, 0.5],
                    'lengths': [0.55, 0.55],},
}

control_params = {
    'mode_choice'         : ['single_mode'], # which mode to assume or mixed
    'belief_weighting'    : False,    # weight the model force by belief
    'model_force'         : False,    # enable direct forwarding of human force
    'model_force_scale'   : 0.6,      # scale the human force model
    'mpc'                 : True,     # enable/disable mpc
    'safety_control'      : False,    # enable/disable the safety controller
    'safety_dist'         : 0.3,      # dist from safety points  damp inc
    'safety_points'       : [[1.08, -0.25, 0.75],
                             [0.80, -0.23, 0.65]], # points where damp inc
    'safe_damping'        : [4000.0], # target safe damping for contact
    'print_control'       : True,}

data_root = getcwd() + '/data/nj60_single_mode/'
data = {'single_mode' : {data_root+'front.bag',
                         data_root+'middle.bag',
                         data_root+'back.bag'}}

gp_params = {
    'num_model_points' : 1000, # number of points to subsample to in GP
    'num_sparse_points': 50,   # 0 = no sparsify, otherwise # of points in Sparse
    'mag_dir'          : False,# convert forces to magnitude and direction
    'trim_thresh'      : 1.0,  # when ||f|| below, remove from GP dataset
    'obs_noise'        : 0.1,  # magnitude of noise to add to observations
    'opt_hyper'        : False,# whether to optimize hyperparams
    'opt_numeric'      : False,# GP hyperparams by {T:Numpy, F:IPOPT}
    'gp_method'        : 'TA', # ME (mean equiv) faster, TA (tay approx) more accurate
    'print_hyper'      : False,# print the fit hyperparams
    'plot'             : True, # plot the fit GPs
    'plot_data'        : True, # plot the fit GPs over training data
    'plot_extension'   : 0.1,  # distance beyond model bounds to plot
    'model_path'       : data_root+'GP_models.pkl', # path to saved models
    'data_path'        : data,
    'hyper'            : {'length_scale' : 1.4e-1, # larger = more smoothing
                          'signal_var'   : 2.7e0,  # == sqrt(sf2)  
                          'noise_var'    : 3.5e0,  # == sqrt(sn2)
                          'mean'         : 0.0,},}

ipopt_options = {  # copied from the GP_MPC library @Kevin 9.4 @Christian: added some options
    'ipopt.linear_solver': 'mumps',           # Linear solver used for NLP
    # Output Options
    'ipopt.print_user_options': 'no',          # Print all options set by the user default 'no'
    'ipopt.print_options_documentation': 'no', # if selected, the algorithm will print the list of all available algorithmic options with some documentation before solving the optimization problem. The default value for this string option is "no"
    'ipopt.print_frequency_iter': 1,           # Determines at which iteration frequency the summarizing iteration output line should be printed. Default: 1
    'ipopt.print_level' : 0,                   # Sets the default verbosity level for console output. The larger this value the more detailed is the output. The valid range for this integer option is 0 ≤ print_level ≤ 12 and its default value is 5.
    'ipopt.file_print_level': 5,               # Verbosity leve for output file. 0 - 12; Default is 5
    'ipopt.print_timing_statistics' : 'no',    # If selected, the program will print the CPU usage (user time) for selected tasks. The default value for this string option is "no".
    # Termination (@Christian: so far all in default)
    'ipopt.tol' : 1e-8,                        # Desired convergence tolerance(relative). (Default is 1e-8)
    'ipopt.max_iter': 3000,                    # Maximum number of iterations (default is 3000)
    'ipopt.max_cpu_time' : 1e6,                # Maximum number of CPU seconds. (Default is 1e6)
    # Barrier Parameter
    'ipopt.mu_strategy': 'adaptive',        # Determines which barrier parameter update strategy is to be used. Default "monotone", can choose 'adaptive' instead
    'ipopt.mu_init' : 0.01,                 # This option determines the initial value for the barrier parameter (mu). It is only relevant in the monotone, Fiacco-McCormick version of the algorithm.
    'ipopt.mu_oracle' : 'probing',
    # NLP
    'ipopt.check_derivatives_for_naninf': 'no',
    'ipopt.bound_relax_factor': 1e-5,                # Factor for initial relaxation of the bounds. Default: 1e-8
    # Warm Start
    'ipopt.warm_start_init_point' : 'yes',
    'ipopt.warm_start_bound_push' : 1e-9,           # default is 0.001
    'ipopt.warm_start_bound_frac' : 1e-9,           # default is 0.001
    'ipopt.warm_start_slack_bound_frac' : 1e-9,     # default is 0.001
    'ipopt.warm_start_slack_bound_push' : 1e-9,     # default is 0.001
    'ipopt.warm_start_mult_bound_push' : 1e-9,      # default is 0.001
    # Restoration Phase
    'ipopt.expect_infeasible_problem': 'no',    # Enable heuristics to quickly detect an infeasible problem. Default 'no'
    'ipopt.expect_infeasible_problem_ctol': 0.0001,     # Threshold for disabling "expect_infeasible_problem" option.
    'ipopt.expect_infeasible_problem_ytol': 1e8,        # Multiplier threshold for activating "expect_infeasible_problem" option.
    'ipopt.start_with_resto': 'no',                     # Tells algorithm to switch to restoration phase in first iteration.
    'ipopt.soft_resto_pderror_reduction_factor': 0.9999,# Required reduction in primal - dual error in the soft restoration phase.
    'ipopt.required_infeasibility_reduction': 0.9,      # Required reduction of infeasibility before leaving restoration phase.
    'ipopt.bound_mult_reset_threshold': 1000,           # Threshold for resetting bound multipliers after the restoration phase.
    'ipopt.constr_mult_reset_threshold': 0,             # Threshold for resetting equality and inequality multipliers after restoration phase.
    'ipopt.evaluate_orig_obj_at_resto_trial': 'yes',    # Determines if the original objective function should be evaluated at restoration phase trial points.

    'ipopt.acceptable_constr_viol_tol' : 1e-8,
    'ipopt.constr_viol_tol' : 1e-6,
    #'ipopt.fixed_variable_treatment' : 'make_constraint',

    #@Kevin 18.6.21: added b/c the lag hess calc was dominating calc time
    #'ipopt.hessian_approximation' : 'limited-memory',
    'print_time' : False,
    'verbose' : False,
    #'expand' : True
}
# path + filename
data_root = getcwd()
path = data_root+'/config/gp_params.yaml'
yaml_save(path, gp_params)

path = data_root+'/config/mpc_params.yaml'
yaml_save(path, mpc_params)

path = data_root+'/config/control_params.yaml'
yaml_save(path, control_params)

path = data_root+'/config/mode_detector_params.yaml'
yaml_save(path, mode_detector_params)

path = data_root+'/config/ipopt_params.yaml'
yaml_save(path, ipopt_options)


#load dict from yamlfile
gp_params_loaded = yaml_load(path)

#Check if it works :)
print("gp_params before save to YAML file:")
for key in gp_params:
    print(key, '   has value  ', gp_params[key])

print("\n \ngp_params loaded from YAML file:")
for key in gp_params_loaded:
    print(key, '   has value  ', gp_params_loaded[key])
