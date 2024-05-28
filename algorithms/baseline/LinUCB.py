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
            # Compute UCB for each action
            ucb_max = float('-inf')
            a_max = self.action_set[0]
            self.selected_action_idx = 0
            for idx, a in enumerate(self.action_set):
                ucb = (a @ self.theta_est) + (beta_sqrt * np.sqrt(a @ (self.Vinv @ a)))
                if ucb > ucb_max:
                    ucb_max = ucb
                    a_max = a
                    self.selected_action_idx = idx  
            return a_max
    
        ##### L2 unit ball case
        else:
            # TODO
            return None