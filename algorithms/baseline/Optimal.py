
import numpy as np

from ..LinearBandits import *
from utils.utils import *


############################################################
#                                                          #
#                     Optimal model                        #
#                                                          #
############################################################


class Optimal(LinearBandit):
    """ 
        Optimal model
            Knows the true model parameter so it recommends the best action possible each time
    """
    def recommend(self):
        ##### Finite set case
        if self.finite_set:
            if self.action_set_dict is not None:
                self.selected_action_idx = self.target
            elif self.rewards is not None:
                self.selected_action_idx = np.argmax(self.rewards)
            else:
                # Get the action with highest value
                self.selected_action_idx = np.argmax(self.action_set @ self.theta)
            return self.action_set[self.selected_action_idx]

        ##### L2 unit ball case
        else:
            best_action, _ = project_L2_Ball(self.theta)
            return best_action
