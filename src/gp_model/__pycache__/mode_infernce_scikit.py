"""
 * This  material is the exclusive property of Fraunhofer IPK and must be
 * returned   to   Fraunhofer IPK,  Robotics  and Automation, immediately
 * upon   request.   This  material  and  the  information illustrated or
 * contained herein  may  not be used, reproduced, stored in  a retrieval
 * system, or transmitted in whole or in part in  any  way -  electronic,
 * mechanical,  photocopying, recording, or otherwise,  without the prior
 * written consent of Fraunhofer IPK.
 * 
 *                       All Rights Reserved
 *                        Copyright (C)  2016
 *                           Fraunhofer IPK
 * 
 **/
//============================================================================
// Name        : mode_inference.py
// Author      : Kevin Haninger
// Version     : v1.0
// Copyright   : Fraunhofer IPK
// Description : Set up a ROS node for publishing state and change in adm gains.
// Created on  : Dec 16 2020
// Last change : 
//============================================================================
"""

import numpy as np

    def get_prediction(state):
        # Return a predicted observation, marginalized over the current belief of mode.
        return None

    def model_likelihood_evaluate(model, obs, state):
       obs_hat, obs_sigma = self.models[mode].predict(state, return_std = True)
       err = obs - obs_hat
       lik = -np.log(np.det(obs_sigma)) - np.matmul(np.matmul(np.transpose(err),np.inverse(obs_sigma)),err)
       return lik

    def init_models_labelled(obs, state, mode):
       kern = DotProduct() + ConstantKernel()
       if mode not in self.models:
           self.num_modes += 1
           print('Adding new dynamic mode {}'.format(mode))
        self.models[mode] = GaussianProcessRegressor(kernel = kern, random_state = 0).fit(state, obs)
        print('Fit model for mode {} with log likelihood {}'.format(mode, self.models[mode].log_marginal_likelihood()))

        with open(self.model_path, 'w') as f:
            pickle.dump(self.models, f)

    def init_models_dataset(path, mode):
        # Load rosbags (or other data format) and initialize models
        bag = rosbag.Bag(path)
        num_obs = bag.get_message_count('robot_state')
        state = np.zeros(12,num_obs)
        obs = np.zeros(6,num_obs)

        current_obs = 0
        for _, msg, _ in bag.read_messages(topics=['robot_state']):
            state[:,current_obs] = [msg.position, msg.velocity]
            obs[:,current_obs] = msg.effort
            current_obs += 1

        init_models_labelled(obs, state, mode)
