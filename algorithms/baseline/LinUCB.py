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
    def recommend_loop(self):
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
        
    def recommend_matmul(self):
        beta_sqrt = ((np.sqrt(self.lam) * self.m2) + (self.sigma * np.sqrt((- 2 * np.log10(self.delta)) + 
                               (self.d * np.log10(1 + ((self.t * self.L**2)/(self.lam*self.d)))))) ) * self.scale

        # Compute intermediate results
        A_Theta = self.action_set @ self.theta_est
        A_Vinv_A = np.einsum('ij,ij->i', self.action_set, self.action_set @ self.Vinv.T)  
        sqrt_A_V_A = np.sqrt(A_Vinv_A)     

        # Compute UCB values for all actions and find maximum
        ucb_values = A_Theta + beta_sqrt * sqrt_A_V_A  
        ucb_values = np.round(ucb_values, decimals=5)
        ucb_max_idx = np.argmax(ucb_values)
        ucb_max = ucb_values[ucb_max_idx]

        a_max = self.action_set[ucb_max_idx]
        self.selected_action_idx = ucb_max_idx
  
        return a_max

    def recommend(self):
        a = self.recommend_matmul()
        # a = self.recommend_loop()
        return a