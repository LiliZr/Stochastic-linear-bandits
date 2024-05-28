import numpy as np
from scipy.linalg import sqrtm

from ..LinearBandits import *
from utils.utils import *

############################################################
#                                                          #
#       Linear Bandit With Regression General Class        #
#                                                          #
############################################################

class LinearRegression(LinearBandit):
    def update_V(self, a_t):
        """
            Update V with given action
                Compute : V_t = V_0 + ∑ action_t ⋅ action_t.T
        """
        self.V = self.V + np.outer(a_t, a_t)


    def update_ar(self, a_t, r_t):
        """
            Update ar cumulative sum
                Compute : ar_t = ∑ action_t ⋅ reward_t
        """
        self.ar = self.ar + (a_t * r_t)

    def init_run(self, T, init_Vinv=True):
        super().init_run(T)
        ## Params used in regression
        # Init inverse of covariance matrix
        if init_Vinv:
            self.Vinv = (1/self.lam) * np.identity(self.d) if self.lam != 0 else np.zeros((self.V.shape))
        # Init product action x reward
        self.ar = np.zeros(self.d)
        # Estimator of the parameter theta
        self.theta_est = np.zeros(self.d)

    def update(self, a_t, r_t):
        # Update parameters
        a_t = self.action_set[self.selected_action_idx]
        self.update_ar(a_t, r_t)
        inv_sherman_morrison(self.Vinv, a_t)

        # Linear regression
        self.theta_est = self.Vinv @ self.ar 


    def recommend(self):
        """
            Recommend over a finite or convex set
        """
        ##### Finite set case
        if self.finite_set:
            # Get the action with highest value
            self.selected_action_idx = np.argmax(self.action_set @ self.theta_est)
            return self.action_set[self.selected_action_idx]

        ##### L2 unit ball case 
        else: 
            best_action, _ = project_L2_Ball(self.theta_est)
            return best_action
