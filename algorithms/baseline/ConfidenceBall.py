
import numpy as np
import scipy

from algorithms.baseline.LinearRegression import *

from ..LinearBandits import *
from utils.utils import *

############################################################
#                                                          #
#                   ConfidenceBall1                        #
#                                                          #
############################################################


class ConfidenceBall1(LinearRegression):
    """ 
        LinREL class recommending over a finite set or convex set
    """
    def recommend(self):
        beta_sqrt = ((np.sqrt(self.lam) * self.m2) + (self.sigma * np.sqrt((- 2 * np.log10(self.delta)) + 
                               (self.d * np.log10(1 + ((self.t * self.L**2)/(self.lam*self.d)))))) ) * self.scale
        ext = extrema(self.Vinv, beta_sqrt * np.sqrt(self.d))

        ##### Finite set case
        if self.finite_set:
            # Compute UCB for each action
            ucb = self.action_set @ (ext + self.theta_est).T
            # Get the action with highest UCB
            self.selected_action_idx, _ = np.unravel_index(np.argmax(ucb, axis=None), ucb.shape)
            return self.action_set[self.selected_action_idx]

        ##### L2 unit ball case
        else:
            opt_value = float("-inf")
            best_action = None

            # Get the action with the highest UCB
            for u in ext:
                ucb = self.theta_est + u
                action, value = project_L2_Ball(ucb)
                if value > opt_value:
                    opt_value = value
                    best_action = action
            return best_action
    