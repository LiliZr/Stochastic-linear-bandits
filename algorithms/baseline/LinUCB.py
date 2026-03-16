import numpy as np


from algorithms.baseline.LinearRegression import *
from ..LinearBandits import *
from utils.utils import *

############################################################
#                                                          #
#                        LinUCB                            #
#                                                          #
############################################################


class LinUCB(LinearRegression):
    """ 
        LinUCB class recommending over a finite set or convex set
    """
    def recommend(self):
        beta_sqrt = ((np.sqrt(self.lam) * self.m2) + (self.sigma * np.sqrt((- 2 * np.log10(self.delta)) + 
                               (self.d * np.log10(1 + ((self.t * self.L**2)/(self.lam*self.d)))))) ) * self.scale
        ##### Finite set case
        if self.finite_set:
            # Compute intermediate results
            ucb_values = self.action_set @ self.theta_est    

            # Compute UCB values for all actions and find maximum
            ucb_values += beta_sqrt * np.sqrt(np.einsum('ij,ij->i', self.action_set, self.action_set @ self.Vinv.T) )  
            ucb_max_idx = np.argmax(ucb_values)

            a_max = self.action_set[ucb_max_idx]
            self.selected_action_idx = ucb_max_idx
    
            return a_max
    
        ##### L2 unit ball case
        else:
            # TODO
            return None