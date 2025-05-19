# Class for sequential refinement of GPs
#
# Based on: Sinsbeck and Nowack, Sequential design of computer experiments for Bayesian inverse problems, SIAM/ASA JUQ, 2017
#
# Current limitations:
#   constant trend function assumed
#   no noise in GP-regression
# 
# Assumptions: 
#   requires input x to be of size (n_samples x n_inputs)
#   requires output y to be of size (n_samples x n_outputs)

from functools import partial
import numpy as np
from scipy.stats import norm
from UQpy.surrogates import *
from scipy.integrate import quad
from sklearn.metrics import pairwise_distances
import copy

class SequentialGP():
    def __init__(self, GP, x, y, jointX):
        self.x = x
        self.y = y    
        self.GP = GP
        self.jointX = jointX
        for i in range(y.shape[1]):
            self.GP[i].fit(samples=x, values=y[:,i])

    # generate sample in X-domain
    def sample(self, n):
        samples = self.jointX.sample(size=n, rule='latin_hypercube')
        return samples.T
    
    # predict associated to output gp_ind at x
    def predict(self, x, gp_ind):
        gp = self.GP[gp_ind]
        val, std = gp.predict(x.T, return_std=True)
        return val, std

    # add new point to training data
    def append(self, x_new, y_new):
        self.x = np.append(self.x, x_new, axis=0)
        self.y = np.append(self.y, y_new, axis=0)
        for i in range(self.y.shape[1]):
            self.GP[i].fit(self.x, self.y, optimizations_number=None, hyperparameters=self.GP[i].hyperparameters.tolist())
    
    
    def Bayes_risk(self, xp, z, sigma_eps, size_X_sample):
        np.random.seed(42)
        n_output = len(self.GP)

        X_samples = self.jointX.sample(size = size_X_sample)
        X_samples = X_samples.T

        x_extended = np.append(self.x, np.array([xp]), axis=0)

        # precompute mu, sigma at xp and x
        mu_xp = np.empty((1, self.y.shape[1]))
        sigma_xp = np.empty((1, self.y.shape[1]))
        mu_x = np.empty((X_samples.shape[0], self.y.shape[1]))
        sigma_x = np.empty((X_samples.shape[0], self.y.shape[1]))
        for i, gp in enumerate(self.GP):
            mu_xp[0,i], sigma_xp[0,i] = gp.predict(xp,return_std=True)
            
            mu_tmp, sigma_tmp = gp.predict(X_samples,return_std=True)
            mu_x[:,i] = mu_tmp.reshape(1,-1) 
            sigma_x[:,i] = sigma_tmp.reshape(1,-1)

        # some routines for obtaining an affine linear updated mean
        def assemble_K(gp,x1,x2):
            distances = pairwise_distances(x1/gp.hyperparameters[0], x2/gp.hyperparameters[1], metric='euclidean')
            K = gp.hyperparameters[2]**2*np.exp(-0.5*distances**2)
            return K
        
        def condition_GP_affine(gp,X_samples,xp):                
            xE = np.append(self.x, np.array([xp]), axis=0)
            K_x = assemble_K(gp,X_samples,x_extended)
            K_p = assemble_K(gp,xp.reshape(1,-1),self.x)
            K_pp = assemble_K(gp,xp.reshape(1,-1),xp.reshape(1,-1))

            A = gp.K
            B = K_p.transpose()
            C = K_p
            D = K_pp

            S = D - np.dot(C,np.linalg.solve(A,B))
            Ainv_f = np.linalg.solve(A,self.y)
            a_vals = np.dot(K_x,np.append(Ainv_f + np.dot(np.linalg.solve(A,B)*S**(-1),np.dot(C,Ainv_f)), -S**(-1)*np.dot(C,Ainv_f), axis=0))
            b_vals = np.dot(K_x,np.append(-np.linalg.solve(A,B)*S**(-1),S**(-1), axis = 0))
            return a_vals, b_vals

        # loop over sample points and output quantities
        vals = np.ones((X_samples.shape[0],1))
        for ind_GP in range(n_output):
            
            # condition GP
            gp = copy.copy(self.GP[ind_GP])
            a_vals, b_vals = condition_GP_affine(gp,X_samples,xp)
            a_vals = a_vals[:,ind_GP]
            
            ## extract sigma_post - the value of y is arbitrary here
            y_extended = np.append(self.y[:,ind_GP], [20], axis=0)    
            gp.fit(samples=x_extended, values=y_extended, optimizations_number=1, hyperparameters=gp.hyperparameters.tolist())
            _, sigma_post_x = gp.predict(X_samples, return_std=True)            

            for ind in range(X_samples.shape[0]):
        	    
                Sigma2 = sigma_post_x[ind]**2 + sigma_eps[ind_GP]**2
                
                numerator = np.exp(-(a_vals[ind] - z[ind_GP] + b_vals[ind]*mu_xp[0,ind_GP])**2/(Sigma2 + 2*b_vals[ind]**2*sigma_xp[0,ind_GP]**2))
                denominator = 2*np.pi*Sigma2 * np.sqrt(1 + (2*b_vals[ind]**2*sigma_xp[0,ind_GP]**2)/Sigma2)
                expec_var_part_two_x = numerator / denominator

                numerator = np.exp(-(a_vals[ind] - z[ind_GP] + b_vals[ind]*mu_xp[0,ind_GP])**2/(2*sigma_post_x[ind]**2 + 2*b_vals[ind]**2*sigma_xp[0,ind_GP]**2 + sigma_eps[ind_GP]**2))
                denominator = 2*np.pi*sigma_xp[0,ind_GP]*sigma_post_x[ind]*sigma_eps[ind_GP]**2*np.sqrt(1/sigma_post_x[ind]**2 + 2/sigma_eps[ind_GP]**2)*np.sqrt(1/sigma_xp[0,ind_GP]**2 + 2*b_vals[ind]**2/(2*sigma_post_x[ind]**2 + sigma_eps[ind_GP]**2))
                expec_var_part_one_x = numerator / denominator

                vals[ind] *= (expec_var_part_one_x - expec_var_part_two_x)

        result = np.mean(vals)    
        
        return result