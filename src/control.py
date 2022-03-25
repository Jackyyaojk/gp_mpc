#!/usr/bin/env python3
# Copyright (c) 2021 Kevin Haninger

# Python libs
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# ROS imports
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Custom classes
from mode_inference import mode_detector
from gp_wrapper import gp_model
from gp_dynamics import GPDynamics
from MPC_GP3 import MPC
from helper_fns import *

# Evaluation
import itertools


class mpc_impedance_control():
    '''
    This class produces commands to change the impedance of a robot according to
    a model of the environment/human and some control policy
    '''
    def __init__(self):
        # Loading config files
        self.mpc_params = yaml_load('config/mpc_params.yaml')
        self.mode_detector_params = yaml_load('config/mode_detector_params.yaml')
        self.gp_params = yaml_load('config/gp_params.yaml')

        self.rotation = self.mpc_params['enable_rotation']
        self.state_dim = 3 if not self.rotation else 6  # range of state
        self.obs_dim   = 3 if not self.rotation else 6  # range of forces

        np.random.seed(0)
        np.set_printoptions(formatter={'float': '{: 6.2f}'.format})

        # Set up or load gp models
        self.gp_models = gp_model(self.gp_params, rotation = self.rotation)
        self.models, self.modes = self.gp_models.load_models()
        self.mode_detector = mode_detector(self.modes, self.models,
                                           params = self.mode_detector_params)

        # Set up local state
        self.recieved_robot_state = False
        self.impedance_params = {'M': np.array([10, 10, 10, 2, 2, 2], dtype=float),
                                 'B': np.array([500, 500, 500, 10, 10, 10], dtype=float),
                                 'K': np.zeros(self.state_dim),
                                 'Fd': np.zeros(self.state_dim)}

        # Init Dynamic Modes
        self.gp_dynamics_dict = { mode: GPDynamics(N_p = self.obs_dim,
                                                   mpc_params = self.mpc_params,
                                                   gp = self.models[mode] )\
                                  for mode in self.modes }
        # Init MPC
        self.mpc = MPC( N_p = self.obs_dim,
                        mpc_params = self.mpc_params,
                        gp_dynamics_dict = self.gp_dynamics_dict )

        # Init ROS
        self.sub         = rospy.Subscriber('robot_state',
                                            JointState, self.update_state, queue_size=1)
        self.sub_params  = rospy.Subscriber('impedance_gains',
                                            JointState, self.update_params, queue_size=1)
        self.pub_belief  = rospy.Publisher ('belief',
                                            Float64MultiArray, queue_size = 1)
        self.pub_control = rospy.Publisher ('delta_impedance_gains',
                                            JointState, queue_size = 1)
        if self.mpc_params['sim']: self.pub_imp = rospy.Publisher('impedance_gains_sim', JointState, queue_size = 1)

        # Init animation
        if self.mpc_params['live_plot'] or self.mpc_params['save_plot']:  self.animate_init()

        # Performance profiling
        self.control_time = time.time()
        self.timelist = []


    # Callback function when state message recieved
    def update_state(self, msg):
        self.recieved_robot_state = True
        try:
            self.state = msg_to_state(msg)
            self.obs   = msg_to_obs(msg)
        except:
            print("Error loading ROS message in update_state")
        if np.linalg.norm(self.obs[:3]) > self.mode_detector_params['min_force']:
            bel_arr = self.mode_detector.update_belief(self.obs[:self.obs_dim], self.state[self.state_dim])
            if self.pub_belief:
                msg_belief = Float64MultiArray(data = bel_arr)
                self.pub_belief.publish(msg_belief)

    # Callback function when impedance parameters message recieved
    def update_params(self, msg):
        if self.mpc_params['sim']: return
        try:
            self.impedance_params['K']  = np.array(msg.position)[:self.state_dim]
            self.impedance_params['B']  = np.array(msg.velocity)[:self.state_dim]
            self.impedance_params['M']  = np.array(msg.effort)[:self.state_dim]
            self.impedance_params['Fd'] = np.array(msg.effort)[6:6+self.state_dim]
        except:
            print("Error loading ROS message in update_params")

    # Main control loop, update belief, do MPC calc, send out the updated params
    def control(self, send_zeros = False):
        if not self.recieved_robot_state: return
        prstr = ''
        if self.mode_detector_params['print_belief']:
            prstr += 'Bel '
            for mode in self.modes:
                prstr += mode + ':' + '{: 6.3f}'.format(self.mode_detector.bel[mode])+' | '

        if not rospy.is_shutdown():
            des_force = np.zeros(self.obs_dim)
            if not send_zeros:
                start = time.time()
                u_opt_traj = self.mpc.solve(self.state, self.mode_detector.bel,
                                            self.impedance_params['M'], self.impedance_params['B'])
                self.timelist.append((time.time() - start))
                des_force = u_opt_traj[:self.obs_dim,0]

                if self.mpc_params['print_control']:
                    prstr += ' Fd  {} | '.format(des_force)
                    prstr += 'mpc {: 6.3f} | '.format(time.time()-start)
                    if self.mpc_params['opti_MBK']:
                        prstr += '\n Delta_M {} | Delta_B {} '\
                                  .format(-self.mpc.mbk_traj[:self.state_dim], -self.mpc.mbk_traj[:self.state_dim])
                        prstr += '\n M {} | B {} | '.format(self.impedance_params['M'][:self.state_dim],
                                                            self.impedance_params['B'][:self.state_dim] )
                    prstr += 'total {: 6.3f}'.format(time.time()-self.control_time)
                    self.control_time = time.time()
                    print(prstr)

                if self.mpc_params['sim']:
                    if self.mpc_params['opti_MBK']:
                        self.impedance_params['M'][:self.state_dim] += np.array(self.mpc.mbk_traj[:self.state_dim])
                        self.impedance_params['B'][:self.state_dim] += np.array(self.mpc.mbk_traj[self.state_dim:])
                    self.build_and_publish(des_force = des_force,
                                           #d_mass = self.impedance_params['M'][:self.state_dim],
                                           #d_damp = self.impedance_params['B'][:self.state_dim])
                                           d_mass = self.mpc.mbk_traj[:self.state_dim],
                                           d_damp = self.mpc.mbk_traj[self.state_dim:])
                    return

            self.build_and_publish(des_force = des_force)

    # Build and publish the ROS messages
    def build_and_publish(self, des_force = None, des_damp = None, d_damp = None, d_mass = None):
            msg_control = JointState()
            msg_control.position = np.zeros(6)
            msg_control.velocity = np.zeros(6)
            msg_control.effort = np.zeros(12)
            msg_control.header.stamp = rospy.Time.now()
            des_force = 0.5*(des_force-self.impedance_params['Fd'][:self.obs_dim]) #to handle oscilation due to delay
            msg_control.effort[6:6+self.obs_dim] = des_force
            if des_damp is not None:
                msg_control.velocity = 0.7*(des_damp - self.impedance_params['B'][:self.obs_dim])
            if d_damp is not None :
                msg_control.velocity[:self.state_dim] = d_damp
            if d_mass is not None:
                msg_control.effort[:self.state_dim] = d_mass
            if not rospy.is_shutdown():
                self.pub_control.publish(msg_control)
                if self.mpc_params['sim']:
                    msg_imp = JointState()
                    msg_imp.position = np.zeros(6)
                    msg_imp.velocity = np.zeros(6)
                    msg_imp.effort = np.zeros(12)
                    msg_imp.header.stamp = rospy.Time.now()

                    msg_imp.position = np.array(self.impedance_params['K'])
                    msg_imp.velocity = np.array(self.impedance_params['B'])
                    msg_imp.effort[:self.state_dim] = np.array(self.impedance_params['M'][:self.state_dim])
                    msg_imp.effort[6:6+self.state_dim] = np.array(self.impedance_params['Fd'][:self.state_dim])
                    self.pub_imp.publish(msg_imp)

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
        if self.mpc_params['save_plot']:
            plt.savefig('animation_frames/'+str(self.fig_number).zfill(3))
            self.fig_number += 1

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

    def shutdown(self):
        print('Shutting down controller')
        self.control(send_zeros = True)
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

if __name__ == '__main__':
    rospy.init_node('mpc_impedance_control')
    node = mpc_impedance_control()

    # Set shutdown to be executed when ROS exits
    rospy.on_shutdown(node.shutdown)
    rospy.sleep(1e-1) # Sleep so ROS can init

    while not rospy.is_shutdown():
        node.control()
        if node.mpc_params['live_plot'] or node.mpc_params['save_plot']:
            node.animate_update()
        rospy.sleep(1e-8) # Sleep so ROS subscribers can update
