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

Code associated with `https://arxiv.org/abs/2110.12433`

Quickstart:
 - collect rosbags with force info, put into directory, adjust gp_params.yaml to point to it
 - python3 -m gp_mpc.control

Validation: python3 -m analysis.validate [--path data/polish_flat/ -f 2.bag 3.bag 4.bag]


## Software Requirements
 - [ ] CasADi
 - [ ] HSL linear solvers (recommended!)
 - [ ] ROS


