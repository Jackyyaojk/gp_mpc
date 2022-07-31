# Copyright (c) 2021 Christian Hegeler, Kevin Haninger

import casadi as ca
import numpy as np

from copy import deepcopy

from .helper_fns import *
from .gp_model import GP

class GPDynamics:
    '''
    This class wraps generates the dynamics and cost function, given a GP model
    '''
    def __init__(self, N_p, mpc_params, gp):
        self.__N_p = N_p   # Num positions  (3 or 6 depending on with or without rotation)
        self.__dt = mpc_params['dt']
        self.__Q_pos = mpc_params['Q_pos']    # stagecost on positions
        self.__Q_vel = mpc_params['Q_vel']    # stagecost on velocities
        self.__R = mpc_params['R']            # stagecost on input, linear
        self.__Rr = mpc_params['Rr']          # stagecost on input, rotational
        self.__S = mpc_params['S']            # stagecost on covariance
        self.__H = mpc_params['H']            # stagecost on human forces
        self.__H_jt = mpc_params['H_jt']      # stagecost on human joint torques
        if 'H_pow' in mpc_params.keys(): self.__H_pow = mpc_params['H_pow']
        self.__I = mpc_params['I']
        self.human_kin = mpc_params['human_kin']

        self.mpc_params = mpc_params
        self.__gp = gp
        self.dec_vars = {}

    def build_dec_vars(self):
        # Defining inputs for the dynamics function f(x,u)
        w = {}
        N_p = self.__N_p
        cov_flag = self.mpc_params['state_cov']
        u = ca.SX.sym('u', N_p)
        x = ca.SX.sym('x', 2*N_p+cov_flag*2*N_p)
        x_next = ca.SX(2*N_p+cov_flag*2*N_p,1) # state at next time step
        x_pos_cov = ca.diag(x[2*N_p:3*N_p]) if self.mpc_params['state_cov'] else []
        hum_jts = ca.SX.sym('hums', 4) if self.mpc_params['opti_hum_jts'] else []

        # Defining parameters
        imp_mass = ca.SX.sym('imp_mass',N_p)
        imp_damp = ca.SX.sym('imp_damp', N_p)
        init_pose = ca.SX.sym('init_pose', 6) # Initial robot pose, position + rotation vector
        return u, x, x_next, x_pos_cov, hum_jts, imp_mass, imp_damp, init_pose


   # def integrator(self, x, u, f_mu, imp_params, x_next):

    # Defines a mass-spring-damper system with a GP force model
    def MDS_system(self):
        # Shortening for ergonomics
        N_p = self.__N_p    # Num of positions in system
        dt = self.__dt      # Time step for discretization

        u, x, x_next, x_pos_cov, hum_jts, imp_mass, imp_damp, init_pose = self.build_dec_vars()

        x_w = compliance_to_world(init_pose, x[:self.__N_p])
        f_mu, f_cov = self.__gp.predict(x=x_w[:self.__N_p], cov=[], fast = self.mpc_params['simplify_cov'])

        # For each DOF, apply the dynamics update
        f_cov_array = []
        for i in range(N_p):
            #kn = dt*K_k/M_k
            if self.mpc_params['integrator']  == 'implicit':
                bn = imp_mass[i]/(imp_mass[i]+dt*imp_damp[i])
            else:
                bn = ca.exp(-dt*imp_damp[i]/imp_mass[i])

            # Velocity first b/c that's needed for semi-implicit
            #x_next[i+N_p] =  bn*x[i+N_p]+dt/(imp_mass[i]+dt*imp_damp[i])*(-f_mu[i]+u[i]) # -kn*x[i]
            x_next[i+N_p] =  bn*x[i+N_p]+dt/imp_mass[i]*(-f_mu[i]+u[i]) # -kn*x[i]
            

            # Position
            if self.mpc_params['integrator'] == 'explicit':
                x_next[i] = x[i]+dt*x[i+N_p]
            elif self.mpc_params['integrator'] == 'implicit':
                x_next[i] = x[i]+dt*x_next[i+N_p] #x_next[i+N_p]
            else:
                print('Integrator {} not supported'.format(self.mpc_params['integrator']))

            # Update state covariance
            if self.mpc_params['state_cov']:
                x_next[i+2*N_p] = x[i+2*N_p]+dt*dt*x[i+3*N_p] # cov pos
                f_cov_tmp = f_cov[0] if self.mpc_params['simplify_cov'] else f_cov[i]
                x_next[i+3*N_p] = bn**2*x[i+3*N_p]+10*(dt/imp_mass[i])**2*f_cov_tmp
                if i < 3: f_cov_array.append(f_cov_tmp)

        # Define stagecost L, note control costs happen in main MPC problem as control shared btwn modes
        L = self.__Q_vel*ca.sumsqr(x_next[N_p:2*N_p]) + self.__R*ca.sumsqr(u[:3]) + self.__I*ca.sum1(f_cov)
        if N_p == 6: L += self.__Rr*ca.sumsqr(u[3:6])
        if self.mpc_params['state_cov']: L += self.__S*ca.sum1(x_next[2*N_p:])

        # Add cost for total force or error from expected human force
        L += self.__H*ca.sumsqr(f_mu+u[:N_p]) if self.mpc_params['match_human_force'] else self.__H*ca.sumsqr(f_mu) 
        if self.__H_pow is not None:
            L += self.__H_pow*f_mu.T@x_next[N_p:2*N_p]

        # Add human kinematic model + cost, if modelling human kinematics
        f_joints = []
        hum_kin_opti = []
        if self.human_kin and self.__H_jt:
            shoulder_pos = self.mpc_params['human_kin']['center']
            if self.__H_jt is not 0.0:
                f_joints, h_jac = self.human_joint_torques_cart(ca.vertcat(x_w[:3], init_pose[3:]),
                                                                shoulder_pos,
                                                                f_mu)
                #imp_damp* ca.fabs(x_next[N_p:2*N_p])+
                power_comp = imp_damp*(0.1*np.ones((3,1)))
                power_world = compliance_to_world(init_pose, power_comp)
                f_joints = h_jac.T@power_world

                L += self.__H_jt*ca.sumsqr(ca.sumsqr(f_joints))
                #L += self.__H_jt*ca.sumsqr(imp_damp*(ca.sumsqr(x_w[:3]-self.mpc_params['human_kin']['center'])-0.1))
                #jt_spd = ca.pinv(h_jac)@x_next[N_p:2*N_p]
                #L += 0.2*self.__H_pow*ca.sumsqr(f_joints*jt_spd)

        dynamics = ca.Function('F_int', [x, u, init_pose, imp_mass, imp_damp],\
                               [x_next, L, f_mu, f_cov, f_joints], \
                               ['x', 'u', 'init_pose',  'imp_mass', 'imp_damp'], \
                               ['xf', 'st_cost', 'hum_force_cart', 'f_cov', 'cost_debug'] )
             # {"jit":True, "jit_options":{'flags':'-O3'}}) # Can JIT the dynamics, less helpful for typical problem
        return dynamics

    def gp_grad(self, x):
        return self.__gp.grad(x)

    def split_cost_function(self, x_traj, u_traj = None):
    #returns a string of the contribution of various parts of the cost function
        types = ['pos', 'vel', 'x_cov', 'f', 'f_cov']
        if u_traj is not None: types += 'u'
        cost_total = {typ:0.0 for typ in types}
        Np = self.__N_p
        for x in x_traj.T:
            cost_total['pos'] += self.__Q_pos*x[:Np].T @ x[:Np]
            cost_total['vel'] += self.__Q_vel*x[Np:2*Np].T @ x[Np:2*Np]
            cost_total['x_cov'] += self.__S*x[2*Np:].T @ x[2*Np:]
            f_mu, f_cov = self.__gp.predict_fast(x=x[:Np])
            cost_total['f'] += self.__H*f_mu.T@f_mu
            cost_total['f_cov'] += self.__I*np.trace(f_cov)
        for u in u_traj:
            cost_total['u'] += self.__R*np.sum(u**2)
        return cost_total

    def human_IK(self, x_ee, shoulder_pos):
        # IK solution for the human joint angles, assuming the interior/exterior
        # rotation of human is 0
        if not self.human_kin: return [0,0,0,0]
        l1 = self.human_kin['lengths'][0]
        l2 = self.human_kin['lengths'][1]
        rel_pos = [x_ee[0]-shoulder_pos[0], x_ee[1]-shoulder_pos[1], x_ee[2]-shoulder_pos[2]]

        dist_2 = ca.fmin(ca.sumsqr(ca.vertcat(*rel_pos)), l1**2+l2**2-1e-4)
        q4 = np.pi-ca.acos((l1**2 + l2**2 - dist_2)/(2*l1*l2))
        q1 = ca.atan2(rel_pos[1], rel_pos[0])
        q2 = ca.asin(rel_pos[2]/ca.sqrt(dist_2))-q4/2
        q3 = 0.0
        return [q1, q2, q3, q4]

    def human_jac(self):
        jts = ca.SX.sym('q',4)
        wrist_pos, _  = self.human_FK(jts, self.human_kin['center'])
        human_jac = ca.jacobian(ca.vertcat(*wrist_pos), jts)
        human_jac_fn = ca.Function('human_jac', [jts], [human_jac])
        return human_jac_fn

    def human_joint_torques_cart(self, x_ee, shoulder_pos, F_comp):
        # Forces are in compliance frame, x_ee in world coords
        F_world = force_comp_to_world(x_ee, F_comp)
        if not hasattr(self, 'human_jac_fn'):
            self.human_jac_fn = self.human_jac()
        jts = self.human_IK(x_ee, shoulder_pos = shoulder_pos)
        jac = self.human_jac_fn(ca.horzcat(*jts))

        return jac.T@F_world, jac

    def human_joint_torques_joint(self, x_ee, jts, F_comp):
        # Where human_kin is shoulder position and joint coords.
        F_world = force_comp_to_world(x_ee, F_comp)
        if not hasattr(self, 'human_jac_fn'):
            self.human_jac_fn = self.human_jac()
        jac = self.human_jac_fn(jts)
        return jac.T@F_world

    def human_FK(self, jts, shoulder_pos):
        if not self.human_kin: return [0,0,0,0]
        #jts has 4 elements:
         # angle about world x axis
         # angle about world y axis
         # internal/external rotation (about upper arm joint)
         # elbow pos
        # zero pose:
         # l1 along x0 and l2 along x4 if -q4 is applied about y3 in  http://web.mit.edu/2.05/www/Handout/HO2.PDF
        q1=jts[0]
        q2=-jts[1]
        q3=jts[2]
        q4=jts[3]
        l1 = self.human_kin['lengths'][0]
        l2 = self.human_kin['lengths'][1]
        wrist_pos = deepcopy(shoulder_pos)
        wrist_pos[0] += l1*ca.cos(q1)*ca.cos(q2)
        wrist_pos[1] += l1*ca.sin(q1)*ca.cos(q2)
        wrist_pos[2] += -l1*ca.sin(q2)

        elbow_pos = deepcopy(wrist_pos)

        wrist_pos[0] += l2*(ca.cos(-q4)*ca.cos(q1)*ca.cos(q2)-ca.sin(-q4)*(ca.cos(q1)*ca.sin(q2)*ca.cos(q3)+ca.sin(q1)*ca.sin(q3)))
        wrist_pos[1] += l2*(ca.cos(-q4)*ca.sin(q1)*ca.cos(q2)-ca.sin(-q4)*(ca.sin(q1)*ca.sin(q2)*ca.cos(q3)-ca.cos(q1)*ca.sin(q3)))
        wrist_pos[2] += l2*(-ca.cos(-q4)*ca.sin(q2)          -ca.sin(-q4)*ca.cos(q2)*ca.cos(q3))
        return wrist_pos, elbow_pos


    def opt_exp_joint_effort(self, x_ee, lam = 50.0):
        from scipy.linalg import solve_sylvester, pinv
        mu, cov = self.__gp.predict(x=x_ee[:self.__N_p], cov=[])
        q = self.human_IK(x_ee[:3], self.mpc_params['human_kin']['center'])
        #print(q)
        jac_fn = self.human_jac()
        jac = jac_fn(ca.horzcat(*q))
        #print(jac)
        jac_inv = pinv(jac)
        #mu = 0.001*ca.DM([10.0, 50.0, 1.0])
        A = (lam*np.eye(3)+mu[:3]@mu[:3].T)@jac_inv.T@jac_inv
        B = cov[:3, :3]
        A = A.full()
        B = B.full()
        Q = lam*np.eye(3)
        #print(A)
        #print(B)
        Imp = solve_sylvester(A, B, Q)
        #print(Imp)
        return Imp


    def plot_exp_joint_effort(self):
        import matplotlib.pyplot as plt

        state_bounds = [np.full(self.__N_p,1e10), np.full(self.__N_p,-1e10)]
        current_range = self.__gp.get_xrange()
        state_bounds = [np.minimum(state_bounds[0],current_range[0]),
                        np.maximum(state_bounds[1],current_range[1])]
        fig, ax = plt.subplots(dpi=300)
        #fig.subplots_adjust(left=-0.11, right=1.05, bottom=0.0, top=1.0)
        exp = 0.2
        x, y = np.meshgrid(np.linspace(state_bounds[0][0]-exp, state_bounds[1][0]+exp, 9),
                           np.linspace(state_bounds[0][1]-exp, state_bounds[1][1]+exp, 9))
        x = x.flatten()
        y = y.flatten()
        avg = self.__gp.get_mean_state()
        meansx = []
        meansy = []
        covs = []
        opt_impx = []
        opt_impy = []
        
        for xi, yi in zip(x,y):
            query_pt = np.concatenate((np.array([xi,yi]), avg[2:]))
            mu, cov = self.__gp.predict(query_pt, cov=[])
            meansx.append(mu[0])
            meansy.append(mu[1])
            opt_imp = self.opt_exp_joint_effort(query_pt)
            opt_impx.append(opt_imp[0,0])
            opt_impy.append(opt_imp[1,1])
            covs.append(0.5*np.linalg.norm(np.diag(cov)))
        #ax.plot(x,y, 'k', marker = 'o', ms = covs, labael = 'Force Cov')
        ax.quiver(x, y, meansx, meansy, color = 'k', alpha = 0.3, label = 'Force Mean')
        ax.quiver(x, y, opt_impx, 0, color = 'b', label = 'Optimized Imp')
        ax.quiver(x, y, 0, opt_impy, color = 'b', label = 'Optimized Imp')

        ax.plot(self.mpc_params['human_kin']['center'][0], self.mpc_params['human_kin']['center'][1],
                'r',  marker = 'o', ms = 10, label = 'human shoulder')
        fig.tight_layout()
        ax.legend()
        plt.show()
