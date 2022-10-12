# GP-MPC Impedance Control

Solve an MPC problem where:
 - **Model:** 
   - Gaussian Processes model forces, possibly over multiple modes (discrete GP models).
   - Impedance dynamics of robot, planning over mean and covariance per mode
   - Basic human kinematic model (fixed shoulder position)
 - **Objective:**
   - Mean, covariance of robot trajectory, force model
   - Expected human forces or joint torques
   - Expected cost or risk-sensitive
 - **Decision variables:**
    - Desired force/torque of impedance controller
    - Impedance parameters
    - Human kinematics (shoulder position, joint angles) 

Repo also includes code for building/fitting GP models from rosbags and visualizing them.

Code associated with `https://arxiv.org/abs/2110.12433` and `https://arxiv.org/abs/2208.07035`


## Software Requirements
 - [ ] CasADi (`python3 -m pip install casadi`)
 - [ ] ROS
 - [ ] [HSL linear solvers](https://github.com/casadi/casadi/wiki/Obtaining-HSL) (recommended!)

# Quickstart
 - [ ] Install dependencies
 - [ ] `git clone --recurse-submodules https://gitlab.cc-asp.fraunhofer.de/hanikevi/gp-mpc-impedance-control`
 - [ ] Collect demonstrations:
   - [ ] `rosbag record -a`
   - [ ] Do demonstrations. We've mostly tested with three demos, fewer/more might be OK.
   - [ ] Adjust `gp_params.yaml` to point to data for each goal/mode
 - [ ] Re-write the ROS interface
   - [ ] Adjust topic names for robot pose and force `control.py::70-76`
   - [ ] `helper_fns.py::msg_to_state` - mapping from robot state ROS message to pose [x, r], where x is in meters and r is rotation vector in rad
   - [ ]   `helper_fns.py::msg_to_obs` mapping from robot ros message to forces [f, t], where f is linear force in robot TCP pose, and t torques
 - [ ] `python3 -m gp_mpc.control`, optionally with arguments, e.g. `--path data/contact_var/` or `--rebuild_gp`
 - [ ] validate your results! Can generate plots with `python3 -m analysis.validate` or `analysis/rosbag_plot.ipynb`

