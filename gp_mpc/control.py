# Copyright (c) 2021 Kevin Haninger

# Python libs
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse

# ROS imports
import rospy
import tf2_ros as tf
import dynamic_reconfigure.client
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker


# Custom classes -- need absolute import b/c of __main__ here.
from gp_mpc.mode_inference import mode_detector
from gp_mpc.gp_wrapper import gp_model
from gp_mpc.gp_dynamics import GPDynamics
from gp_mpc.mpc import MPC
from gp_mpc.helper_fns import *

class mpc_impedance_control():
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

        # Loading config files
        self.mpc_params = yaml_load(path, 'mpc_params.yaml')
        self.mode_detector_params = yaml_load(path, 'mode_detector_params.yaml')

        self.rotation = self.mpc_params['enable_rotation']
        self.state_dim = 3 if not self.rotation else 6  # range of state

        np.set_printoptions(formatter={'float': '{: 7.2f}'.format})

        # Set up or load gp models, GPs will be built if (1) there's no .pkl with the models, or (2) rebuild_gp is true
        self.gp_models = gp_model(path, rotation = self.rotation)
        self.models, self.modes = self.gp_models.load_models(rebuild = rebuild_gp)

        # Updates belief about which mode is active
        self.mode_detector = mode_detector(self.modes, self.models,
                                           params = self.mode_detector_params)

        # Set up robot and mpc state
        self.rob_state = {k:None for k in ('imp_stiff', 'des_pose', 'pose')}
        self.mpc_state = {}

        # Init dynamics for each mode
        self.gp_dynamics_dict = { mode: GPDynamics(N_p = self.state_dim,
                                                   mpc_params = self.mpc_params,
                                                   gp = self.models[mode] )\
                                  for mode in self.modes }

        # Init MPC
        self.mpc = MPC( N_p = self.state_dim,
                        mpc_params = self.mpc_params,
                        gp_dynamics_dict = self.gp_dynamics_dict,
                        path = path )

        # Init ROS
        self.sub_imp_xd  = rospy.Subscriber('cartesian_impedance_example_controller/equilibrium_pose',
                                            PoseStamped, self.update_imp_xd, queue_size=1)
        self.sub_force   = rospy.Subscriber('franka_state_controller/F_ext',
                                            WrenchStamped, self.update_force, queue_size=1)
        self.tf_buffer = tf.Buffer()
        self.listener = tf.TransformListener(self.tf_buffer)
        self.pub_belief  = rospy.Publisher ('belief',
                                            Float64MultiArray, queue_size = 1)
        self.pub_imp_xd = rospy.Publisher ('equilibrium_pose_mpc',
                                            PoseStamped, queue_size = 1)

        self.pub_traj = rospy.Publisher('mpc_traj', Path, queue_size=1)

        self.pub_imp_xd_marker = rospy.Publisher ('equilibrium_pose_marker',
                                                  Marker, queue_size = 1)
        if not self.mpc_params['sim']: self.par_client = dynamic_reconfigure.client.Client("/cartesian_impedance_example_controller/dynamic_reconfigure_compliance_param_node")

        # If this is in simulation; i.e. a bag file is being used, add pub which integrates the delta_impedance_gains
        if self.mpc_params['sim']:
            print("mpc_param sim is set; integrating imp params in ctrl")
            self.pub_imp = rospy.Publisher('impedance_gains_sim',
                                           JointState, queue_size = 1)

        # Init animation
        if self.mpc_params['live_plot']:  self.animate_init()
        rospy.sleep(0.2)
        self.init_orientation = self.tf_buffer.lookup_transform('panda_link0', 'panda_link8', rospy.Time(0)).transform.rotation

        # Performance profiling
        self.control_time = time.time()
        self.timelist = []


    # Callback function when force message recieved
    def update_force(self, msg):
        obs = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

        # If multiple modes and there is external force, update the mode belief
        if len(self.modes) > 1 and np.linalg.norm(obs[:3]) > self.mode_detector_params['min_force'] and self.rob_state['pose']:
            bel_arr = self.mode_detector.update_belief(obs[:self.state_dim], self.rob_state['pose'][:self.state_dim])
            if self.pub_belief:
                msg_belief = Float64MultiArray(data = bel_arr)
                self.pub_belief.publish(msg_belief)

    def update_state_async(self):
        state_msg = self.tf_buffer.lookup_transform('panda_link0', 'panda_link8', rospy.Time(0))
        state = msg_to_state(state_msg)
        self.rob_state['pose'] = state
        if self.mpc_params['sim']:
            if self.mpc_state:
                self.rob_state['imp_stiff'] = self.mpc_state['imp_stiff']
        else:
            imp_pars = self.par_client.get_configuration()
            self.rob_state['imp_stiff'] = np.array((imp_pars['translational_stiffness_x'],
                                                    imp_pars['translational_stiffness_y'],
                                                    imp_pars['translational_stiffness_z']))

    # Callback function when impedance parameters message recieved
    def update_imp_xd(self, msg):
        if self.mpc_params['sim'] and self.mpc_state:
            self.rob_state['des_pose'] = self.mpc_state['des_pose']
        else:
            self.rob_state['des_pose']  = np.array([msg.pose.position.x,
                                                    msg.pose.position.y,
                                                    msg.pose.position.z])

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
        self.control_time = time.time()
        print(prstr)

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

        if self.mpc_params['print_control']: self.print_results()

        self.build_and_publish()

    # Build and publish the ROS messages
    def build_and_publish(self, send_zeros = False):
        if send_zeros:
            print("Sending zero on all delta impedance gains and desired force")
            return

        msg_imp_xd = get_empty_pose()
        msg_imp_xd.pose.position.x = self.mpc_state["des_pose"][0]
        msg_imp_xd.pose.position.y = self.mpc_state["des_pose"][1]
        msg_imp_xd.pose.position.z = self.mpc_state["des_pose"][2]
        msg_imp_xd.pose.orientation = self.init_orientation

        self.publish_traj()

        if not rospy.is_shutdown():
            self.pub_imp_xd.publish(msg_imp_xd)
            if self.mpc_params['sim']:
                msg_imp = get_empty_jointstate()
                msg_imp.position = np.array(self.mpc_state.get('imp_stiff'))
                self.pub_imp.publish(msg_imp)
            else:
                res = self.par_client.update_configuration({'translational_stiffness_x':self.mpc_state['imp_stiff'][0],
                                                            'translational_stiffness_y':self.mpc_state['imp_stiff'][1],
                                                            'translational_stiffness_z':self.mpc_state['imp_stiff'][2]})

    def publish_traj(self):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = 'panda_link8'
        for traj_pt in self.mpc_state['x_peg1']:
            traj_pt_pose = get_empty_pose(frame_id = 'panda_link8')
            traj_pt_pose.pose.position.x = traj_pt[0]
            traj_pt_pose.pose.position.y = traj_pt[1]
            traj_pt_pose.pose.position.z = traj_pt[2]
            path.poses.append(traj_pt_pose)
        self.pub_traj.publish(path)

    def animate_update(self):
        # Plot planned trajectories
        for mode in self.modes:
            traj_comp = self.mpc.x_traj[mode]
            position_traj = comp_traj_to_world(self.state, traj_comp)
            self.lines[mode].set_data_3d(position_traj[:3,:])
            self.ax.draw_artist(self.lines[mode])
            if self.rotation:
                self.plot_coord(mode)
        # Plot EE position
        self.robot_ee.set_data_3d(self.state[:3])
        self.ax.draw_artist(self.robot_ee)

        # Plot Impedance
        rot = rotvec_to_rotation(self.state[3:6])
        for i in range(3):
            offset = 1e-4*rot[:,i]*self.impedance_params['B'][i]
            self.robot_imp[i].set_data_3d([[self.state[0], self.state[0]+offset[0]],
                                           [self.state[1], self.state[1]+offset[1]],
                                           [self.state[2], self.state[2]+offset[2]]])
            self.ax.draw_artist(self.robot_imp[i])
        # Plot human arm
        if 'human_kin' in self.mpc_params:
            gp = self.gp_dynamics_dict[self.modes[0]]
            if self.mpc_params['opti_hum_shoulder']:
                human_center = self.mpc.hum_shoulder
                wrist, elbow = gp.human_FK(self.mpc.hum_joints, self.mpc.hum_shoulder)
            else:
                human_center = np.array(self.mpc_params['human_kin']['center'])
                wrist, elbow = gp.human_FK(gp.human_IK(position_traj[:3,0], human_center), human_center)
            self.human_arm.set_data_3d(np.vstack((human_center, elbow, wrist)).T)
            self.ax.draw_artist(self.human_arm)
            self.human_shoulder.set_data_3d(human_center)
            self.ax.draw_artist(self.human_shoulder)
        self.fig.canvas.flush_events()


    def animate_init(self):
        # 3D figure / attaching 3D / formatting
        plt.ion()
        self.fig_number = 0
        self.fig = plt.figure()
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=-0.11, right=1.11, bottom=0.0, top=1)
        self.coord_frames = {}
        plt.show(block=False)
        plt.pause(1e-5)
        self.ax = self.fig.add_subplot(projection="3d")
        init_pose = np.array([0.8, 0, 0.5, 0, 0, 0])
        ax = self.ax
        ax.view_init(elev = 20, azim = 12)
        ax_bnd = 0.35
        ax.set_xlim3d([init_pose[0]-ax_bnd, init_pose[0]+ax_bnd])
        ax.set_xlabel('X')
        ax.set_ylim3d([init_pose[1]-ax_bnd, init_pose[1]+ax_bnd])
        ax.set_ylabel('Y')
        ax.set_zlim3d([init_pose[2]-ax_bnd, init_pose[2]+ax_bnd])
        ax.set_zlabel('Z')
        ax.set_title('Trajectory Plot')
        self.lines = {}
        colors = list('rbgk')
        # Set up empty line objects to change later
        for mode in self.modes:
            self.lines[mode] = ax.plot(0, 0, 0, marker = 'o', color = colors.pop())[0]
            if self.rotation:
                self.plot_coord(mode)
        self.robot_ee = ax.plot(0,0,0, marker = 'o', ms=10, c='k')[0]
        self.robot_imp  = [ax.plot(0,0,0, color = 'k', linewidth=3)[0],
                           ax.plot(0,0,0, color = 'k', linewidth=3)[0],
                           ax.plot(0,0,0, color = 'k', linewidth=3)[0]]

        if 'human_kin' in self.mpc_params:
           hc = self.mpc_params['human_kin']['center']
           self.human_shoulder = ax.plot(0,0,0, c = 'r', marker = 'o', ms = 10)[0]
           self.human_arm = ax.plot(0,0,0, color='red', linewidth=3)[0]
        self.fig.canvas.draw()
        return self.lines

    def plot_coord(self, mode):
        sc = 0.08
        if mode not in self.coord_frames.keys():
            self.coord_frames[mode] = []
            for i in range(self.mpc_params['mpc_pts']+1):
                plt_handles_x = []
                plt_handles_x.append(self.ax.plot([0, 0],[0, 0], [0, 0],'r')[0])
                plt_handles_x.append(self.ax.plot(0,0,0,'g')[0])
                plt_handles_x.append(self.ax.plot(0,0,0,'b')[0])
                self.coord_frames[mode].append(plt_handles_x)
        else:
            init_orient = rotvec_to_rotation(self.state[3:])
            world_pos_traj = comp_traj_to_world(self.state, self.mpc.x_traj[mode])
            for i in range(len(world_pos_traj[0])):
                x = world_pos_traj[:,i]
                rot = rotvec_to_rotation(x[3:6])
                self.coord_frames[mode][i][0].set_data_3d([x[0], x[0]+sc*rot[0,0]],[x[1], x[1]+sc*rot[1,0]],[x[2], x[2]+sc*rot[2,0]])
                self.ax.draw_artist(self.coord_frames[mode][i][0])
                self.coord_frames[mode][i][1].set_data_3d([x[0], x[0]+sc*rot[0,1]],[x[1], x[1]+sc*rot[1,1]],[x[2], x[2]+sc*rot[2,1]])
                self.ax.draw_artist(self.coord_frames[mode][i][1])
                self.coord_frames[mode][i][2].set_data_3d([x[0], x[0]+sc*rot[0,2]],[x[1], x[1]+sc*rot[1,2]],[x[2], x[2]+sc*rot[2,2]])
                self.ax.draw_artist(self.coord_frames[mode][i][2])

    # Callback function executed by ROS when node is shutdown
    def shutdown(self):
        print('Shutting down controller')
        self.build_and_publish(send_zeros = True)
        self.get_results()

    # Time evaluation
    def get_results(self):
        if self.timelist:
            t_stats = np.array(self.timelist)
            t_cold = t_stats[0]
            t_mean = np.mean(t_stats[1:])
            t_min = min(t_stats[1:])
            t_max = max(t_stats[1:])
            print("Cold Start: {}, Mean: {}, Min: {}, Max: {}".format(t_cold, t_mean, t_min, t_max))

def start_node(args):
    rospy.init_node('mpc_impedance_control')
    node = mpc_impedance_control(args)

    # Set shutdown to be executed when ROS exits
    rospy.on_shutdown(node.shutdown)
    rospy.sleep(1e-1) # Sleep so ROS can init

    while not rospy.is_shutdown():
        node.update_state_async()
        node.control()
        if node.mpc_params['live_plot']: node.animate_update()
        rospy.sleep(1e-8) # Sleep so ROS subscribers can update

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/pih/", help="Root folder for data & config")
    parser.add_argument("--rebuild_gp", default=False, action='store_true',
                        help="Force a new Gaussian Process to build")
    args = parser.parse_args()
    start_node(args)
