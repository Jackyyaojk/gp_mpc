# Copyright (c) Kevin Haninger

import numpy as np
from gp_mpc.gp_model import GP

class mode_detector():
    '''
    This class allows the current dynamic mode of a system, \eta, to be detected
    given conditional probabilities p(y|x,\eta).  This conditional distribution
    can be taught from batch data
    '''
    def __init__(self, modes, models, params):
        self.params = params
        self.modes = modes
        self.models = models
        self.log_bel = {mode:np.log(1.0/len(self.modes)) for mode in self.modes}
        self.bel = {mode:1.0/len(self.modes) for mode in self.modes}

    def update_belief(self, obs, state):
        normalization_constant = 0
        for mode in self.modes:
            if self.params['use_direction']:
                lik = self.predict_llik_dir(state, obs, mode)
                if np.isnan(lik): lik = 0
                self.log_bel[mode] += lik/self.params['bel_smoothing']
            else:
                self.log_bel[mode] += self.predict_llik(state, obs, mode)/self.params['bel_smoothing']
            self.log_bel[mode] = max(self.log_bel[mode], self.params['bel_floor'])
            if np.isnan(self.log_bel[mode]):
                print("NaN in belief! That's.... not good")
                self.log_bel[mode] = self.params['bel_floor']
            normalization_constant += np.exp(self.log_bel[mode])
            #print("log bel before norm {}: {}".format(mode, self.log_bel[mode]))
        normalization_constant += np.exp(self.params['anomaly_lik'])
        self.log_bel = {mode:self.log_bel[mode]-np.log(normalization_constant) for mode in self.modes}
        self.bel = {mode:np.exp(self.log_bel[mode]) for mode in self.modes}
        return [np.exp(self.log_bel[mode].__float__()) for mode in self.modes]
 
    def predict_llik(self, x, y, mode):
        """ Give log likelihod of a model having that observation
        """
        mean, var = self.models[mode].predict(x, [], [])
        lik_floor = self.params['lik_floor']
        if np.any(var<0.0): print("GP covariance has negative components")
        err  = y-mean
        lik = []
        for i in range(len(y)): # covariane is diagonal, so we decompose the total likelihood
            if not np.isnan(var[i,i]) and not np.isnan(err[i]):
                lik.append( -0.5*np.log(var[i,i])-err[i]**2/(np.sqrt(var[i,i])))
                #lik.append( -np.log(var[i,i])-err[i]**2/(var[i,i]))
        lik = np.array(lik)
        if lik_floor: lik[np.where(lik<lik_floor)] = lik_floor
        if self.params['print_dof_lik']: print(lik)
        return np.sum(lik)

    def predict_llik_dir(self, x, y, mode):
        """ Give log likelihod of a model having that observation
        """
        mean, var = self.models[mode].predict_fast(x)
        pred_mag = np.linalg.norm(mean)
        real_mag = np.linalg.norm(y)
        pred_normed = mean/pred_mag
        real_normed = y/real_mag

        if np.any(var<0.0): print("GP covariance has negative components")

        if pred_mag > 1e-9: # Numerical issues if pred is near 0
            dir_sim = np.log(0.5*np.dot(np.squeeze(pred_normed), real_normed)+0.501) #Similarity in direction
        else:
            dir_sim = -3

        dir_sim *= real_mag/20.0 # weight the similarity less when the magnitude is smaller
        mag_sim = np.abs((pred_mag-real_mag)/real_mag)
        log_var = 0.1*np.sum(np.log(np.abs(var)))
        #print('dir: {}, mag: {}, log cov: {}, mode: {}'.format(dir_sim, mag_sim, log_var, mode))
        return dir_sim-log_var


