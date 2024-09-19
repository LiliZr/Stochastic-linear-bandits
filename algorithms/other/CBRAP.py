
from algorithms.baseline.LinearRegression import *


class LinearRegression_CBRAP(LinearRegression):
    def __init__(self, theta, lam=0.0001, sigma=0.1, delta=0.01, 
                 scale=0.01, action_set=None, seed=48,
                 k=10, nb_samples=1000, seed_proj=1):
        """
        """
        super().__init__(theta, lam, sigma, delta, scale, action_set, seed)
        self.nb_samples = nb_samples # Samples L2 Ball
        self.seed = seed * seed_proj
        # Params for generating embedding matrix
        self.k = k
        self.M = None         # Matrix embedding
        print(f'- CBRAP... k = {self.k}, {lam}, {scale}')


    def init_run(self, T):
        # Generate embedding Matrix if not given
        if self.M is None:
            rng = np.random.RandomState(self.seed)
            self.M = rng.normal(0, np.sqrt(1 / self.k), size = (self.k, self.d))
            self.d = self.k

        
        # Project action set in embedding space
        if self.action_set is None:
            self.action_set = sample_uniform_d_sphere(self.nb_samples, self.theta.shape[0]-1)
        
        if self.action_set_dict is None:
            self.action_set_proj = self.action_set @ self.M.T
        

        # Initialization of general structures
        super().init_run(T)

    def update_action_set(self):
        updated = super().update_action_set()
        if updated:
            self.action_set_proj = self.action_set @ self.M.T

    def generate_reward(self, a_t):
        _, a_t = a_t
        return super().generate_reward(a_t)
        
    def update(self, a_t, r_t):
        # Update parameters
        a_proj_t, a_t = a_t
        self.update_ar(a_proj_t, r_t)
        inv_sherman_morrison(self.Vinv, a_proj_t)

        # Linear regression
        self.theta_est = self.Vinv @ self.ar 

############################################################
#                                                          #
#                        CBRAP                            #
#                                                          #
############################################################
    
class CBRAP(LinearRegression_CBRAP):
    """ 
        CBRAP class recommending over a finite set
    """
    def recommend_loop(self):
        """
            Initial Implementation using for loop
        """
        beta = self.scale * ((self.sigma * np.sqrt(self.d * np.log10((1 + self.t) / self.delta))) + np.sqrt(self.lam) + 1)
        # Compute UCB for each action
        ucb_max = float('-inf')
        a_max = None
        a_proj_max = None
        for idx, (a_proj, a) in enumerate(zip(self.action_set_proj, self.action_set)):
            # Compute UCB for each action
            ucb = (a_proj @ self.theta_est) + (beta * np.sqrt(a_proj @ (self.Vinv @ a_proj)))
            if ucb > ucb_max:
                ucb_max = ucb
                a_proj_max = a_proj
                a_max = a   
                self.selected_action_idx = idx 
        return a_proj_max, a_max
    
    def recommend_matmul(self):
        """
            Implementation using matrix multiplication with einsum 
        """
        beta = self.scale * ((self.sigma * np.sqrt(self.d * np.log10((1 + self.t) / self.delta))) + np.sqrt(self.lam) + 1)
        # Compute UCB for each action
        A_Theta = self.action_set_proj @ self.theta_est
        # Line below more efficient than : np.diagonal (self.action_set @ Vinv @ self.action_set^T)             
        A_Vinv_A = np.einsum('ij,ij->i', self.action_set_proj,  self.action_set_proj @ self.Vinv.T)  # get diagonal of (A @ Vinv @ A^T) Of size n 

        sqrt_A_Vinv_A = np.sqrt(A_Vinv_A)  
        ucb_values = A_Theta + beta * sqrt_A_Vinv_A

        # find the maximum UCB and corresponding index
        ucb_max_idx = np.argmax(ucb_values)
        ucb_max = ucb_values[ucb_max_idx]

        # Rretrieve the corresponding action and projection
        a_proj_max = self.action_set_proj[ucb_max_idx]
        a_max = self.action_set[ucb_max_idx]
        self.selected_action_idx = ucb_max_idx
        return a_proj_max, a_max

    def recommend(self):
        a = self.recommend_matmul()
        # a = self.recommend_loop()
        return a