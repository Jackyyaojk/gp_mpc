# Copyright (c) Kevin Haninger

# ROS imports
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import rosbag

# Python libs
import numpy as np
from os.path import isfile
import pickle

# Custom classes
from gp_mpc import GP
from gp_sparse import gp_sparse
from helper_fns import msg_to_state, msg_to_obs, force_comp_to_world

class gp_model():
    '''
    This class wraps the GP class from gp_mpc adding modes, data loading, and plotting
    '''
    def __init__(self, gp_params, rotation = False, try_reload = True):
        self.rotation = rotation
        self.state_dim = 3+3*rotation

        self.state  = {}
        self.obs    = {}
        self.models = {}

        self.gp_params = gp_params
        self.validate_params()

    def validate_params(self):
        self.modes = list(self.gp_params['data_path'].keys())
        if self.gp_params['num_sparse_points'] is not 0:
            print('Sparse GPs: ensuring baseline points is at least 500')
            self.gp_params['num_model_points'] = max(self.gp_params['num_model_points'],500)

        g_p = self.gp_params['hyper']
        g_p['length_scale'] *= np.ones((self.state_dim, 3))
        g_p['signal_var'] *= np.ones((3,1))
        g_p['noise_var'] *= np.ones((3,1))
        g_p['mean'] *= np.ones((self.state_dim,1))
        if self.rotation:
            g_p['length_scale'] = np.append(g_p['length_scale'],\
                    np.full((self.state_dim, 3), self.gp_params['hyper_rot']['length_scale']),1)
            g_p['signal_var'] = np.append(g_p['signal_var'],\
                    np.full((3,1), self.gp_params['hyper_rot']['signal_var']),0)
            g_p['noise_var'] = np.append(g_p['noise_var'],\
                    np.full((3,1), self.gp_params['hyper_rot']['noise_var']),0)

    def load_models(self, try_reload = True):
        if try_reload and isfile(self.gp_params['model_path']):
            with open(self.gp_params['model_path'], 'rb') as f:
                self.models = pickle.load(f)
                self.modes = list(self.models.keys())
                print('Loaded models for {}'.format(self.modes))
        else:
            for mode in self.modes:
                self.load_data(mode)
                self.build_model(mode)
                self.validate_model(mode)
                with open(self.gp_params['model_path'], 'wb') as f:
                    pickle.dump(self.models, f)

        return self.models, self.modes

    def load_data(self, mode):
        # Load rosbags
        paths = self.gp_params['data_path'][mode]
        print("Loading mode {} with {}".format(mode, paths))
        trimmed_msgs = self.load_bags(paths)

        # Sub-sampling
        indices = np.linspace(0, len(trimmed_msgs)-1, self.gp_params['num_model_points']).astype(int)
        subsampled_msgs = [trimmed_msgs[i] for i in indices]

        # Processing msgs to state/obs
        self.state[mode] = np.array([msg_to_state(msg) for msg in subsampled_msgs])
        self.obs[mode]   = np.array([msg_to_obs(msg)   for msg in subsampled_msgs])
        self.obs[mode]  += self.gp_params['obs_noise']*np.random.randn(*np.array(self.obs[mode]).shape)

        # If desired, make sparse GP with synthetic points
        if self.gp_params['num_sparse_points'] is not 0:
            print("Sparsifying model!")
            self.sparsify(mode)

    def load_bags(self, paths):
        trimmed_msgs = []
        for path in paths:
            bag = rosbag.Bag(path)
            topic_name = '/robot_state'
            num_obs = bag.get_message_count(topic_name)
            print('Loading ros bag {}  with {} msgs'.format(path, num_obs))
            if num_obs == 0:
                print("No messages on /robot_state, checking w/o slash")
                topic_name = 'robot_state'
                num_obs = bag.get_message_count(topic_name)
                print('Loading ros bag {}  with {} msgs'.format(path, num_obs))

            t_first = 1e24
            t_last = 0
            # Finding the first and last messages (rosbags not guarnateed to be in order)
            for _, msg, t_ros in bag.read_messages(topics=[topic_name]):
                t = t_ros.to_sec()
                if t < t_first:
                    msg_first = msg
                    t_first = t
                if t > t_last:
                    msg_last = msg
                    t_last = t
            pos_first = np.array(msg_first.position)
            pos_last = np.array(msg_last.position)

            for _, msg, _ in bag.read_messages(topics=[topic_name]):
                close_to_first = np.linalg.norm(np.array(msg.position[:6])-pos_first) \
                    < self.gp_params['trim_thresh_init']
                no_force = np.linalg.norm(np.array(msg.effort[:3])) < self.gp_params['trim_thresh_force']
                workspace = np.array(msg.position[2]<self.gp_params['trim_workspace'])
                if not no_force and not close_to_first and workspace:
                    trimmed_msgs.append(msg)
        print('Total of {} messages after trim'.format(len(trimmed_msgs)))
        return trimmed_msgs

    def build_model(self, mode):
        # Build the GP for obs/state for the given mode
        self.models[mode] = GP(self.state[mode][:,:self.state_dim],
                               self.obs[mode][:, :self.state_dim],
                               hyper = self.gp_params['hyper'],
                               opt_hyper = self.gp_params['opt_hyper'],
                               normalize = False,
                               gp_method = self.gp_params['gp_method'],
                               fast_axis = self.gp_params['simplify_cov_axis'])
        if(self.gp_params['print_hyper']): self.models[mode].print_hyper_parameters()
        print("Built model {} with data dim {}".format(mode, self.state[mode][:,:self.state_dim].shape))
    def validate_model(self, mode):
        if self.gp_params['plot']:
            self.plot_models()
            if self.rotation: self.plot_models(rot = True)
        if self.gp_params['plot_data']:
            self.plot_data()
            if self.rotation:  self.plot_data(rot = True)
        if self.gp_params['plot_linear']:
            self.plot_linear()
        print('GP Likelihood: {}'.format(self.models[mode].log_lik()))

    def sparsify(self, mode):
        gp_s = gp_sparse(self.gp_params, self.state_dim, self.state_dim)
        obs_s, state_s, _ = gp_s.opti(self.obs[mode], self.state[mod])
        return obs_s, state_s

    def plot_models(self, rot = False):
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt

        state_bounds = [np.full(self.state_dim,1e10), np.full(self.state_dim,-1e10)]
        for mode in self.modes:
            current_range = self.models[mode].get_xrange()
            state_bounds = [np.minimum(state_bounds[0],current_range[0]),
                            np.maximum(state_bounds[1],current_range[1])]
        fig = plt.figure(dpi=200)
        ax = fig.gca(projection='3d')
        fig.tight_layout()
        fig.subplots_adjust(left=-0.11, right=1.05, bottom=0.0, top=1.0)

        fig2 = plt.figure(dpi=200)
        ax2 = fig2.gca(projection='3d')
        max_force = np.zeros((3,1))
        min_cov = np.full((3,1), 1e10)
        max_cov = np.zeros((3,1))
        exp = self.gp_params['plot_extension']
        off = 0 if not rot else 3
        x, y, z = np.meshgrid(np.linspace(state_bounds[0][0+off]-exp, state_bounds[1][0+off]+exp, 5),
                              np.linspace(state_bounds[0][1+off]-exp, state_bounds[1][1+off]+exp, 5),
                              np.linspace(state_bounds[0][2+off]-exp, state_bounds[1][2+off]+exp, 5))
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        offset = 0.0
        colors = ['r','g','b']
        for mode in self.modes:
            means = []
            covs   = []
            c = colors.pop()
            test_pt = np.mean(self.state[mode], axis = 0)
            for xi, yi, zi in zip(x,y,z):
                test_pt[off:off+3] = np.array([xi,yi,zi])
                mu, cov = self.models[mode].predict(test_pt[:self.state_dim])
                mu = mu[off:off+3]
                min_cov = np.minimum(cov[off:off+3], min_cov)
                max_cov = np.maximum(cov[off:off+3], max_cov)
                max_force = np.maximum(max_force, np.abs(mu))
                mu = force_comp_to_world(test_pt, mu)
                means.append(mu.full())
                if np.any(np.diag(cov)<0): print("Test point had neg cov! {}".format(cov))
                covs.append(0.5*np.linalg.norm(np.diag(cov)))
            means = np.array(means).squeeze()
            ax.quiver(x, y, z, means[:,0], means[:,1], means[:,2], length=0.002, color = c)
            ax2.scatter(x+offset, y+offset, z+offset, s=covs, color = c)
            offset += 0.02
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax.set_title('GP for {}, rotation: {}'.format(mode, rot))
        ax2.set_title('GP for {}, rotation: {}'.format(mode, rot))
        plt.show()
        print('Mode: {}  | Rotation: {}'.format(mode, rot))
        print('Max covariance: {}'.format(np.diag(max_cov)))
        print('Min covariance: {}'.format(np.diag(min_cov)))
        print('Max force:      {}'.format(max_force.squeeze()))

    def plot_data(self, rot = False):
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt

        for mode in self.modes:
            fig = plt.figure(dpi=200)
            plt.cla()
            ax = fig.gca(projection='3d')
            fig.tight_layout()
            fig.subplots_adjust(left=-0.11, right=1.05, bottom=0.0, top=1.0)

            X_data, Y_data = self.models[mode].get_data()
            for X, Y in zip(X_data, Y_data):
                X_pert = X+0.004*np.random.randn(1,self.state_dim)
                mu, _  = self.models[mode].predict(X_pert)
                
                if not rot:
                    ax.quiver(X_pert[0,0], X_pert[0,1], X_pert[0,2], mu[0], mu[1], mu[2], length= 0.0015, color = 'b')
                    ax.quiver(X[0], X[1], X[2], Y[0], Y[1], Y[2], length=0.0015, color = 'r')
                else:
                    ax.quiver(X_pert[0,3], X_pert[0,4], X_pert[0,5], mu[3], mu[4], mu[5], length= 0.0015, color = 'b')
                    ax.quiver(X[3], X[4], X[5], Y[3], Y[4], Y[5], length=0.0015, color = 'r')
            plt.legend(['Model', 'Data'])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

    def plot_linear(self, axis = 2):
        import matplotlib.pyplot as plt

        rot = axis >= 3
        state_bounds = [np.full(self.state_dim,1e10), np.full(self.state_dim,-1e10)]
        for mode in self.modes:
            current_range = self.models[mode].get_xrange()
            state_bounds = [np.minimum(state_bounds[0],current_range[0]),
                            np.maximum(state_bounds[1],current_range[1])]
        fig = plt.figure(dpi=300)
        ax = fig.gca()
        fig.tight_layout()

        exp = 0.1
        x = 0.5*(state_bounds[0][0]+state_bounds[1][0])
        y = 0.5*(state_bounds[0][1]+state_bounds[1][1])
        z = np.linspace(state_bounds[0][2]-exp, state_bounds[1][2]+exp, 35)

        z = z.flatten()
        colors = ['r','g','b']
        for mode in self.modes:
            meansz = []
            covs   = []
            avg = self.models[mode].get_mean_state() 
            c = colors.pop()
            for zi in z:
                mu, cov = self.models[mode].predict(np.concatenate((np.array([x,y,zi]), avg[3:])))
                meansz.append(mu[2])
                covs.append(0.5*np.linalg.norm(np.diag(cov)))
            plt.plot(-z, meansz, color = c, label = mode)
            meansz = np.array(meansz)
            covs = np.array(covs)
            ax.fill_between(-z, (meansz-covs), (meansz+covs), color=c, alpha=.25)

            X_data, Y_data = self.models[mode].get_data()
            plt.plot(-X_data[:,2], Y_data[:,2], '.', color=c, alpha=.25)
        plt.xlabel('Z position (m)')
        plt.ylabel('Z force (N)')
        plt.legend()
        plt.grid(True)
        plt.show()


