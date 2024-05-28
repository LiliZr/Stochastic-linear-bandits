from algorithms.fjlt.LinearRegression_FJLT import *


############################################################
#                                                          #
#                        LinUCB                            #
#                                                          #
############################################################
    
class LinUCB_FJLT(LinearRegression_FJLT):
    """ 
        LinUCB class recommending over a finite set
    """
    def recommend(self):
        beta_sqrt = ((np.sqrt(self.lam) * self.m2) + (self.sigma * np.sqrt((- 2 * np.log10(self.delta)) + 
                               (self.d * np.log10(1 + ((self.t * self.L**2)/(self.lam*self.d)))))) ) * self.scale

        # Compute UCB for each action
        ucb_max = float('-inf')
        a_max = None
        a_proj_max = None
        for a_proj, a in zip(self.action_set_proj, self.action_set):
            ucb = (a_proj @ self.theta_est) + (beta_sqrt * np.sqrt(a_proj @ (self.Vinv @ a_proj)))
            if ucb > ucb_max:
                ucb_max = ucb
                a_proj_max = a_proj
                a_max = a
                
        return a_proj_max, a_max