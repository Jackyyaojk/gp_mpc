# Copyright (c) 2021 Kevin Haninger, Christian Hegeler

# Python libs
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ROS imports
import rospy
import tf2_ros as tf
import dynamic_reconfigure.client
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray

# Custom classes -- need absolute import b/c of __main__ here.
from gp_mpc.mode_inference import ModeDetector
from gp_mpc.gp_wrapper import GPModel
from gp_mpc.gp_dynamics import GPDynamics
from gp_mpc.mpc import MPC
from gp_mpc.helper_fns import *

class MPCImpedanceControl():
    '''
    This class produces commands to change the impedance of a robot according to
    a model of the environment/human and some control policy
    IN:
      path: directory with data and config files. If no config files at path, defaults loaded from config/
      rebuild_gp: optionally force the GP to be re-built, otherwise it will try to load a .pkl file
    '''
    def __init__(self, args):
        # process args
        path = args.path
        rebuild_gp = args.rebuild_gp
        self.sim = args.sim

        # Loading config files
        self.mpc_params = yaml_load(path, 'mpc_params.yaml')
        self.mode_detector_params = yaml_load(path, 'mode_detector_params.yaml')

        self.rotation = self.mpc_params['enable_rotation']
        self.state_dim = 3 if not self.rotation else 6  # range of state

        np.set_printoptions(formatter={'float': '{: 7.2f}'.format})

        # Set up or load gp models, GPs will be built if (1) there's no .pkl with the models, or (2) rebuild_gp is true
        self.gp_models = GPModel(path, rotation = self.rotation)
        self.models, self.modes = self.gp_models.load_models(rebuild = rebuild_gp)

        # Updates belief about which mode is active
        self.mode_detector = ModeDetector(self.modes, self.models,
                                          params = self.mode_detector_params)

        # Set up robot and mpc state
        self.rob_state = {k:None for k in ('imp_stiff', 'pose')}
        self.rob_state.update(self.mode_detector.get_state())
        self.mpc_state = {}

        # Init dynamics for each mode
        self.gp_dynamics_dict = { mode: GPDynamics(mpc_params = self.mpc_params,
                                                   gp = self.models[mode] )\
                                  for mode in self.modes }

        # Init MPC
        self.mpc = MPC( mpc_params = self.mpc_params,
                        gp_dynamics_dict = self.gp_dynamics_dict,
                        path = path )

        # Init ROS
        self.sub_force   = rospy.Subscriber('franka_state_controller/F_ext',
                                            WrenchStamped, self.update_force, queue_size=1)
        self.tf_buffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)
        self.pub_belief  = rospy.Publisher ('belief', Float64MultiArray, queue_size = 1)
        self.pub_imp_xd = rospy.Publisher ('cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size = 1)
        self.pub_traj = rospy.Publisher('mpc_traj', Path, queue_size=1)

        # If this is in simulation; i.e. a bag file is being used, add pub which integrates the delta_impedance_gains
        if self.sim:
            print("mpc_param sim is set; integrating imp params in ctrl")
            self.pub_imp = rospy.Publisher('impedance_gains_sim', JointState, queue_size = 1)
            self.rob_state['imp_stiff'] = np.array([200, 200, 200])
        else:
            self.par_client = dynamic_reconfigure.client.Client("/cartesian_impedance_example_controller/dynamic_reconfigure_compliance_param_node")
            res = self.par_client.update_configuration({'translational_stiffness_x':40.,
                                                        'translational_stiffness_y':40.,
                                                        'translational_stiffness_z':40.})

        self.init_orientation = self.tf_buffer.lookup_transform('panda_link0', 'panda_EE', rospy.Time(0), rospy.Duration(1)).transform.rotation

        # Performance profiling
        self.timelist = []


    # Callback function when force message recieved
    def update_force(self, msg):
        obs = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

        # If multiple modes and there is external force, update the mode belief
        if len(self.modes) > 1 and np.linalg.norm(obs[:3]) > self.mode_detector_params['min_force']:
            bel_arr = self.mode_detector.update_belief(obs[:self.state_dim], self.rob_state['pose'][:self.state_dim])
            if self.pub_belief:
                msg_belief = Float64MultiArray(data = bel_arr)
                self.pub_belief.publish(msg_belief)

    def update_state_async(self):
        pose_msg = self.tf_buffer.lookup_transform('panda_link0', 'panda_EE', rospy.Time(0), rospy.Duration(0.05))
        self.rob_state['pose'] = msg_to_state(pose_msg)
        if self.sim:
            if self.mpc_state: # assume the desired impedance from MPC is achieved
                self.rob_state['imp_stiff'] = self.mpc_state['imp_stiff']
        else:
            imp_pars = self.par_client.get_configuration()
            self.rob_state['imp_stiff'] = np.array((imp_pars['translational_stiffness_x'],
                                                    imp_pars['translational_stiffness_y'],
                                                    imp_pars['translational_stiffness_z']))

    # Callback function when impedance parameters message recieved
    def update_imp_xd(self, msg):
        if self.sim and self.mpc_state:
            self.rob_state['des_pose'] = self.mpc_state['des_pose']
        else:
            self.rob_state['des_pose']  = np.array([msg.pose.position.x,
                                                    msg.pose.position.y,
                                                    msg.pose.position.z])

    # Main control loop, update belief, do MPC calc, send out the updated params
    def control(self):
        if any(el is None for el in self.rob_state.values()) or rospy.is_shutdown(): return

        # MPC calc
        # Build parameters dictionary for the MPC problem
        params = self.rob_state
        params.update(self.mode_detector.get_state())

        start = time.time()
        self.mpc_state = self.mpc.solve(params)
        self.timelist.append(time.time() - start)

        #print(self.mpc_state['x_peg1'])
        if self.mpc_params['print_control']: self.print_results()

        self.build_and_publish()

    # Build and publish the ROS messages
    def build_and_publish(self):
        des_pose_w = compliance_to_world(self.rob_state['pose'], self.mpc_state['des_pose'], only_position=True)
        msg_imp_xd = get_pose_msg(position = des_pose_w, frame_id='panda_link0')
        msg_imp_xd.pose.orientation = self.init_orientation

        path = self.build_traj_msg()

        if not rospy.is_shutdown():
            self.pub_imp_xd.publish(msg_imp_xd)
            self.pub_traj.publish(path)
            if self.mpc_params['opt_imp']:
                if self.sim:
                    msg_imp = get_empty_jointstate_msg()
                    msg_imp.position = np.array(self.mpc_state.get('imp_stiff'))
                    self.pub_imp.publish(msg_imp)
                else:
                    res = self.par_client.update_configuration({'translational_stiffness_x':self.mpc_state['imp_stiff'][0],
                                                                'translational_stiffness_y':self.mpc_state['imp_stiff'][1],
                                                                'translational_stiffness_z':self.mpc_state['imp_stiff'][2]})

    def build_traj_msg(self):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = 'panda_EE'
        for traj_pt in self.mpc_state['x_left'].T:
            traj_pt_pose = get_pose_msg(position = traj_pt, frame_id = 'panda_EE')
            path.poses.append(traj_pt_pose)
        return path

    def print_results(self):
        prstr = ''  # Print string

        # Belief
        if self.mode_detector_params['print_belief']:
            prstr += 'Bel '
            for mode in self.modes:
                prstr += mode + ':{: 6.3f} | '.format(self.mode_detector.bel[mode])
        for k,v in self.mpc_state.items():
            if k.startswith('x_'): continue # dont want to print trajectory
            prstr += f' {k} {v} | '
        prstr += 'mpc {: 6.3f} | '.format(self.timelist[-1])
        print(prstr)
  
    # Callback function executed by ROS when node is shutdown
    def shutdown(self):
        print('Shutting down controller')
        self.build_and_publish()
        if len(self.timelist) > 1:
            t_stats = np.array(self.timelist)
            print(f"Cold Start: {t_stats[0]}, Mean: {np.mean(t_stats[1:])}, Min: {min(t_stats[1:])}, Max: {max(t_stats[1:])}")

def start_node(args):
    rospy.init_node('mpc_impedance_control')
    node = MPCImpedanceControl(args)

    # Set shutdown to be executed when ROS exits
    rospy.on_shutdown(node.shutdown)
    rospy.sleep(1e-1) # Sleep so ROS can init

    while not rospy.is_shutdown():
        node.update_state_async()
        node.control()
        time.sleep(1e-8) # Sleep so ROS subscribers can update

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/pih/", help="Root folder for data & config")
    parser.add_argument("--rebuild_gp", default=False, action='store_true',
                        help="Force a new Gaussian Process to build")
    parser.add_argument("--sim", default=False, action='store_true',
                        help="For offline tests with rosbags")
    args = parser.parse_args()
    start_node(args)
