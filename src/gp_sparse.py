from os import getcwd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from scipy.linalg import solve_triangular

class gp_sparse():
    '''
    This Class generates artificial GP data using Sparse Gaussian Processes
    X and Y are of dim NxD and NxE, where N is # points, D/E dim of state/obs
    '''
    def __init__(self, gp_params, x_dim, y_dim):
        self.__n_points = gp_params['num_sparse_points']
        self.__y_dim = y_dim
        self.__x_dim = x_dim
        # Get first element of params, assume all params same for all DOF in Y
        self.l = gp_params['hyper']['length_scale'][0][0]
        self.sigma_f = np.sqrt(gp_params['hyper']['signal_var'][0])
        self.sigma_n = np.sqrt(gp_params['hyper']['noise_var'][0])

    def kernel(self, X1, X2):
        """
        Isotropic squared exponential kernel.

        Args:
            X1: Array of m points (m, d).
            X2: Array of n points (n, d).
            theta: kernel parameters (2,)
        """

        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def kernel_diag(self, d):
        """
        Isotropic squared exponential kernel (computes diagonal elements only).
        """
        return np.full(shape=d, fill_value=self.sigma_n ** 2)


    def jitter(self, d, value=1e-6):
        return np.eye(d) * value

    def nlb_fn(self, X, y, sigma_n):
        n = X.shape[0]
        d = X.shape[1]
        r = y.shape[1]

        def nlb(X_m):
            """
            Negative lower bound on log marginal likelihood.
            Args:
                X_m is initial guess to the synthetic dat
                l, sigma_f and sigma_y are GP hyperparams
            """
            X_m = np.reshape(X_m, (-1, d))
            K_mm = self.kernel(X_m, X_m) + self.jitter(X_m.shape[0])
            K_mn = self.kernel(X_m, X)

            L = np.linalg.cholesky(K_mm)  # m x m
            A = solve_triangular(L, K_mn, lower=True) / sigma_n  # m x n
            AAT = A @ A.T  # m x m
            B = np.eye(X_m.shape[0]) + AAT  # m x m
            LB = np.linalg.cholesky(B)  # m x m
            c = solve_triangular(LB, A.dot(y), lower=True) / sigma_n  # m x 1

            # Lower boundary for Kulback-Leiber-Divergenz minimization (see Equation (13) in )
            # see also https://github.com/GPflow/GPflow/blob/1e1de824397c828a47d9eca002251041296c91d4/gpflow/models/sgpr.py
            lb = - n / 2 * r * np.log(2 * np.pi)
            lb -= r * np.sum(np.log(np.diag(LB)))
            lb -= n / 2 * r * np.log(sigma_n ** 2)
            lb -= 0.5 / sigma_n ** 2  *np.sum(y.T.dot(y))
            lb += 0.5 * np.sum(c.T.dot(c))
            lb -= 0.5 / sigma_n ** 2 * np.sum(self.kernel_diag(n))
            lb += 0.5 * np.trace(AAT)

            return -lb

        return nlb

    def phi_opt(self, X_m, X, y, sigma_n):
        """Optimize mu_m and A_m using Equations (11) and (12)."""
        precision = (1.0 / sigma_n ** 2)
        K_mm = self.kernel(X_m, X_m) + self.jitter(X_m.shape[0])
        K_mm_inv = np.linalg.inv(K_mm)
        K_nm = self.kernel(X, X_m)
        K_mn = K_nm.T

        Sigma = np.linalg.inv(K_mm + precision * K_mn @ K_nm)

        mu_m = precision * (K_mm @ Sigma @ K_mn).dot(y)
        A_m = K_mm @ Sigma @ K_mm

        return mu_m, A_m, K_mm_inv

    def opti(self, X, Y):
        n = self.__n_points
        ind = np.random.choice(range(X.shape[0]), n, replace=False)
        x_init = X[ind,:]+0.02*np.random.randn(n,X.shape[1])
        #x_init = np.vstack((np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), n), \
        #                    np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), n), \
        #                    np.linspace(np.min(X[:, 2]), np.max(X[:, 2]) , n))).T
        
        # Run optimization
        print('Start GP_Sparse Optimization (This may take some time)')
        
        res = minimize(fun=self.nlb_fn(X, Y, self.sigma_n), 
                       x0=x_init, method='L-BFGS-B',
                       options={'disp':1, 'gtol': 1e-5, 'maxfun':1e7},
                       jac = '3-point')
        # Optimized kernel parameters and inducing inputs
        X_opt = np.reshape(res.x, (n,-1))
        #
        Y_opt, cov_opt, K_mm_inv = self.phi_opt(X_opt, X, Y, self.sigma_n)
        print('GP_Sparse Optimization finished')
        return X_opt, Y_opt, cov_opt

    def plot(self, X, Y, lb=-30, ub=50, title='Sparse GP plot', color='blue'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        X0, Y0, Z0 = zip(*X)
        U0, V0, W0 = zip(*Y)
        ax.quiver(X0, Y0, Z0, U0, V0, W0, color=color)
        ax.set_xlim([lb, ub])
        ax.set_ylim([lb, ub])
        ax.set_zlim([lb, ub])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        plt.show()

data_root = getcwd() + '/data/nj60_single_mode/'
data = {'single_mode' : {data_root+'front.bag', data_root+'middle.bag', data_root+'back.bag'}}
# data = {'single_mode' : {data_root+'front.bag'}}

obs_range = 3
state_range = 3

gp_params = {
            'num_model_points' : 150,   # number of points in GP
            'num_sparse_points': 60,   # number of points in sparse GP
            'mag_dir'          : False,# convert forces to magnitude and direction
            'trim_thresh'      : 0.1,  # when ||f|| below, remove from GP dataset
            'obs_noise'        : 0.01, # magnitude of noise to add to observations
            'opt_hyper'        : False,# whether to optimize hyperparams
            'opt_numeric'      : False,# GP hyperparams by {T:Numpy, F:IPOPT}
            'print_hyper'      : True, # print the fit hyperparams
            'plot'             : True, # plot the fit GPs
            'plot_data'        : False, # plot the fit GPs over training data
            'plot_extension'   : 0.1,  # distance beyond model bounds to plot
            'model_path'       : data_root+'GP_models.pkl', # path to saved models
            'gp_method'        : 'TA',
            'data_path'        : data,
            'hyper'            : {'length_scale' : 1.5e-1*np.ones((obs_range,
                                                                 state_range)),
                                  'signal_var'   : 1.75e0*np.ones((obs_range,1)), #sigma_f**2, kernel param
                                  'noise_var'    : 3.5e0*np.ones((obs_range,1)), #sigma_n**2, measurement noise
                                  #'signal_var'   : np.expand_dims(np.array([25.0, 1., 1.]),1),  # For mag/dir
                                  #'noise_var'    : np.expand_dims(np.array([10.0, 0.5, 0.5]),1),# For mag/dir
                                  'mean'         : 0.0*np.ones((obs_range,1)),},}

if __name__ == '__main__':
    from gp_model import gp_model

    mode = 'single_mode'

    gp_models = gp_model(gp_params, 3, 3)

    gp_models.init_models()

    data = gp_models.data


    N_subsam = 100
    N_data=len(data)

    #All states and efforts
    state = np.zeros((N_data, state_range))
    obs = np.zeros((N_data, obs_range))
    #Subsampling for ploting
    state_subsampled = np.zeros((N_subsam, state_range))
    obs_subsampled = np.zeros((N_subsam, obs_range))

    #Subsampling
    indices = np.linspace(0, N_data-1, N_subsam).astype(int)

    current_model_point0 = 0
    current_model_point1 = 0
    #fill arrays with data
    for msg in data:
        state[current_model_point0,:] = msg.position[:state_range]
        obs[current_model_point0,:] = msg.effort[:obs_range]
        if current_model_point0 in indices:
            state_subsampled[current_model_point1, :] = msg.position[:state_range]
            obs_subsampled[current_model_point1, :] = msg.effort[:obs_range]
            current_model_point1 +=1
        current_model_point0 +=1


    gps = gp_sparse(gp_params, state_range, obs_range)

    print(state_subsampled)
    pos_opt, eff_opt, cov_opt = gps.opti(state_subsampled, obs_subsampled)
    print(pos_opt)
    #print("Theta parameters are: {}".format(theta_opt))
    gps.plot(pos_opt, eff_opt, color='red')
    gps.plot(state_subsampled, obs_subsampled, color='blue', title="Subsampled_GP")

