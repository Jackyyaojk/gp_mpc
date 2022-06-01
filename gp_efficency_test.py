
# %matplotlib # Uncomment to unlock the _true_ potential of matplotlib (plot outside browser)

from gp_mpc.gp_wrapper import gp_model
from gp_mpc.helper_fns import *

path = "data/rail/" #'data/rail/'  # also data/contact_var/
gp_params = yaml_load(path, 'gp_params.yaml')
gp_params['path'] = path
gp_params['mean_func'] = 'hinge' # zero, const, linear, hinge. zero and const are tested :)
#init of hinge position
gp_params['hinge_position_init'] = [1, 1, 1]
gp_params['linear_init'] = [1, 1, 1]
gp_params['mean_init'] = [100, 100, 1000]

# build GP model
gp_models = gp_model(gp_params, rotation = False)
models, modes = gp_models.load_models(rebuild = True) # rebuild = False loads model specified in gp_params['model_path']

# pick our favorite model
model = models[modes[0]]
test_pt = model.get_mean_state() # get test point from middle of GP data
print("Stiffness at {} is {}".format(test_pt, model.grad(test_pt)[2,2]))  # print derivative of Fz w.r.t. z (i.e. vertical stiffness)