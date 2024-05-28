import numpy as np

from ..LinearBandits import *
from utils.utils import *

############################################################
#                                                          #
#                        Random                            #
#                                                          #
############################################################



class Random(LinearBandit):
    """ 
        Random model
            Recommends actions randomly
    """
    def recommend(self):
        ##### Finite set case
        if self.finite_set:
            # Recommending random action over finite set
            self.selected_action_idx = self.rng.randint(0, self.action_set.shape[0])
            return self.action_set[self.selected_action_idx]

        ##### L2 unit ball case
        else:
            # Recommending random action over L2 unit ball
            theta = self.rng.randn(self.d)
            action, _ = project_L2_Ball(theta)
            return action