import scipy
from algorithms.fjlt.LinearRegression_FJLT import *


############################################################
#                                                          #
#                        ConfidenceBall                    #
#                                                          #
############################################################  


class ConfidenceBall1_FJLT(LinearRegression_FJLT):
    """ 
        ConfidenceBall1 class recommending over a finite set
    """
    def recommend(self):
        beta_sqrt = ((np.sqrt(self.lam) * self.m2) + (self.sigma * np.sqrt((- 2 * np.log10(self.delta)) + 
                               (self.d * np.log10(1 + ((self.t * self.L**2)/(self.lam*self.d)))))) ) * self.scale

        ext = extrema(self.Vinv, beta_sqrt * np.sqrt(self.d))

        ##### Finite set case
        if self.finite_set:
            # Compute UCB for each action
            ucb = self.action_set_proj @ (ext + self.theta_est).T
            # Get the action with highest UCB
            self.selected_action_idx, _ = np.unravel_index(np.argmax(ucb, axis=None), ucb.shape)
            return self.action_set_proj[self.selected_action_idx], self.action_set[self.selected_action_idx]

        ##### L2 unit ball case 
        else:
            opt_value = float("-inf")
            best_action_proj = None

            # Get the action with the highest UCB
            for u in ext:
                ucb = self.theta_est + u
                action_proj, value = project_L2_Ball(ucb, r=np.sqrt((1 + self.eps)))
                if value > opt_value:
                    opt_value = value
                    best_action_proj = action_proj

            idx = idx_nearest_vector(best_action_proj, self.action_set_proj)
            return best_action_proj, self.action_set[idx]
