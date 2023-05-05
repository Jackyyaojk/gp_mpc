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

        self.mpc_params = mpc_params
        self.__gp = gp
        self.dec_vars = {}

    def build_dec_vars(self):
        # Defining inputs for the dynamics function f(x,u)
        w = {}
        N_p = self.__N_p
        cov_flag = self.mpc_params['state_cov']
        
        x = ca.SX.sym('x', 2*N_p+cov_flag*2*N_p)
        x_next = ca.SX(2*N_p+cov_flag*2*N_p,1) # state at next time step
        x_pos_cov = ca.diag(x[2*N_p:3*N_p]) if self.mpc_params['state_cov'] else []
      
        # Defining parameters
        imp_mass = ca.SX.sym('imp_mass', N_p)
        imp_damp = ca.SX.sym('imp_damp', N_p)
        imp_stiff = ca.SX.sym('imp_stiff', N_p)
        des_pose = ca.SX.sym('des_pose', N_p) # Initial robot pose, position + rotation vector
        init_pose = ca.SX.sym('init_pose', 6) # Initial robot pose, position + rotation vector
        return x, x_next, x_pos_cov, imp_mass, imp_damp, imp_stiff, des_pose, init_pose

    # Defines a mass-spring-damper system with a GP force model
    def MDS_system(self):
        # Shortening for ergonomics
        N_p = self.__N_p    # Num of positions in system
        dt = self.__dt      # Time step for discretization
        par = self.mpc_params

        x, x_next, x_pos_cov, imp_mass, imp_damp, imp_stiff, des_pose, init_pose = self.build_dec_vars()

        x_w = compliance_to_world(init_pose, x[:self.__N_p])
        f_mu, f_cov = self.__gp.predict(x=x_w[:self.__N_p], cov=[], fast = self.mpc_params['simplify_cov'])

        # For each DOF, apply the dynamics update
        f_cov_array = []
        for i in range(N_p):          
            bn = imp_mass[i]/(imp_mass[i]+dt*imp_damp[i])
            
            # Velocity first b/c that's needed for semi-implicit
            x_next[i+N_p] =  bn*x[i+N_p]+dt/imp_mass[i]*(-f_mu[i]-imp_stiff[i]*(x_w[i]-des_pose[i]))
            x_next[i] = x[i]+dt*x_next[i+N_p] #x_next[i+N_p]
            
            # Update state covariance
            if self.mpc_params['state_cov']:
                x_next[i+2*N_p] = x[i+2*N_p]+dt*dt*x[i+3*N_p] # cov pos
                f_cov_tmp = f_cov[0] if self.mpc_params['simplify_cov'] else f_cov[i]
                #TODO add the stiffness reduction in cov here
                x_next[i+3*N_p] = bn**2*x[i+3*N_p]+10*(dt/imp_mass[i])**2*f_cov_tmp
                if i < 3: f_cov_array.append(f_cov_tmp)

    
        # Define stage cost, note control costs happen in main MPC problem as control shared btwn modes
        st_cost = par['Q_vel']*ca.sumsqr(x_next[N_p:2*N_p])
        st_cost += par['I']*ca.sum1(f_cov)
        if par['state_cov']: st_cost += par['S']*ca.sum1(x_next[2*N_p])
        #st_cost += self.__H*ca.sumsqr(f_mu+u[:N_p]) if self.mpc_params['match_human_force'] else self.__H*ca.sumsqr(f_mu) 
        if self.mpc_params['state_cov']: st_cost += par['S']*ca.sum1(x_next[2*N_p:])


        dynamics = ca.Function('F_int', [x, des_pose, init_pose, imp_mass, imp_damp, imp_stiff],\
                               [x_next, st_cost, x_w], \
                               ['x', 'des_pose', 'init_pose',  'imp_mass', 'imp_damp', 'imp_stiff'], \
                               ['xf', 'st_cost', 'debug'] )
        return dynamics

    def gp_grad(self, x):
        return self.__gp.grad(x)

