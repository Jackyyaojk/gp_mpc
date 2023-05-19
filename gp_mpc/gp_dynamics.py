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
    def __init__(self, mpc_params, gp):
        self.mpc_params = mpc_params
        self.__gp = gp
        self.state_dim = self.__gp.state_dim
        self.dec_vars = {}

    def build_dec_vars(self):
        # Defining inputs for the dynamics function f(x,u)
        N_p = self.state_dim
        cov_flag = self.mpc_params['state_cov']
        
        x = ca.SX.sym('x', 2*N_p+cov_flag*2*N_p) # state at current time step
        x_next = ca.SX(2*N_p+cov_flag*2*N_p,1)   # state at next time step
        x_pos_cov = ca.diag(x[2*N_p:3*N_p]) if self.mpc_params['state_cov'] else []
       
        # Defining parameters
        imp_mass = ca.SX.sym('imp_mass', N_p)
        imp_damp = ca.SX.sym('imp_damp', N_p)
        imp_stiff = ca.SX.sym('imp_stiff', N_p)
        des_pose = ca.SX.sym('des_pose', N_p) # Rest position of impedance, in TCP coordinates
        init_pose = ca.SX.sym('init_pose', 6) # Initial robot pose in world coordinates

        return x, x_next, x_pos_cov, imp_mass, imp_damp, imp_stiff, des_pose, init_pose

    # Defines a mass-spring-damper system with a GP force model
    def build_dynamics(self):
        # Shortening for ergonomics
        N_p = self.state_dim   # Num of positions in system
        par = self.mpc_params
        dt = par['dt']

        x, x_next, x_pos_cov, imp_mass, imp_damp, imp_stiff, des_pose, init_pose = self.build_dec_vars()

        x_w = compliance_to_world(init_pose, x[:N_p]) # converts trajectory into world coords
        f_mu, f_cov = self.__gp.predict(x=x_w[:N_p], cov=[], fast = self.mpc_params['simplify_cov'])

        # For each DOF, apply the dynamics update        
        for i in range(N_p):          
            # Semi-implicit damping
            bn = imp_mass[i]/(imp_mass[i]+dt*imp_damp[i])

            # Integration
            x_next[i+N_p] = bn*x[i+N_p]+dt/imp_mass[i]*(-f_mu[i]+imp_stiff[i]*(des_pose[i]-x[i]))
            x_next[i] = x[i]+dt*x_next[i+N_p]
            
            # Update state covariance
            if self.mpc_params['state_cov']:
                x_next[i+2*N_p] = x[i+2*N_p]+dt*dt*x[i+3*N_p] # cov pos
                f_cov_tmp = f_cov[0] if self.mpc_params['simplify_cov'] else f_cov[i]
                x_next[i+3*N_p] = bn**2*x[i+3*N_p]+(dt/imp_mass[i])**2*(f_cov_tmp-imp_stiff[i]*x_next[i+2*N_p])
    
        # Define stage cost, note control costs happen in main MPC problem as control shared btwn modes
        st_cost = par['vel_cost']*ca.sumsqr(x_next[N_p:2*N_p])
        st_cost += par['f_cov_cost']*ca.sum1(f_cov)
        if par['state_cov']:
            st_cost += par['pos_cov_cost']*ca.sum1(x_next[2*N_p:3*N_p])
            st_cost += par['vel_cov_cost']*ca.sum1(x_next[3*N_p:4*N_p])
        
        if par['match_gp_force']:
            for i in range(3):
                st_cost += par['f_cost']*ca.sumsqr(f_mu[i]+imp_stiff[i]*(des_pose[i]-x[i]))
        else:
            st_cost += par['f_cost']*ca.sumsqr(f_mu) 


        dynamics = ca.Function('F_int', [x, des_pose, init_pose, imp_mass, imp_damp, imp_stiff],\
                               [x_next, st_cost], \
                               ['x', 'des_pose', 'init_pose',  'imp_mass', 'imp_damp', 'imp_stiff'], \
                               ['x_next', 'st_cost'] )
        return dynamics

    def gp_grad(self, x):
        return self.__gp.grad(x)

