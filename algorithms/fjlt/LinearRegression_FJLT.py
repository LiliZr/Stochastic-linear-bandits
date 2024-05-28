
import tracemalloc
from algorithms.baseline.LinearRegression import *


class LinearRegression_FJLT(LinearRegression):
    def __init__(self, theta, lam=0.0001, sigma=0.1, delta=0.01, 
                 scale=0.01, action_set=None, seed=48, seed_proj=1,
                 k=None, c=0.001, eps=0.01, p=2, nb_samples=1000):
        """
            c: constant used in matrix embedding generation
            eps: distortion parameter in FJLT
            p: kind of embedding we project to
            nb_samples: number of samples from L2 ball in infinite case
        """
        super().__init__(theta, lam, sigma, delta, scale, action_set, seed)
        self.seed = seed * seed_proj
        self.nb_samples = nb_samples # Samples L2 Ball
        # Params for generating embedding matrix
        self.c = c
        self.eps = eps
        self.p = p
        self.k = k
        self.phi = None         # Matrix embedding


    def init_run(self, T):
        # Generate embedding Matrix if not given
        if self.phi is None:
            nb_actions = 0
            if self.action_set_dict is not None:
                for key in self.action_set_dict.keys():
                    nb_actions += self.action_set_dict[key].shape[0]
            else:
                nb_actions = self.action_set.shape[0] if self.action_set is not None else self.nb_samples
            if self.k is None:
                self.k = math.ceil(self.c * self.eps**(-2) * np.log10(nb_actions))
            print(f'- FJLT... K = {self.k}')
            self.phi = generate_phi(self.d, nb_actions, k=self.k, p=self.p, seed=self.seed)
        
        # Project action set in embedding space
        if self.action_set is None:
            self.action_set = sample_uniform_d_sphere(self.nb_samples, self.theta.shape[0]-1)
        
        if self.action_set_dict is None:
            self.action_set_proj = self.action_set @ self.phi.T

        # Update dimension
        self.d = self.k

        # Initialization of general structures
        super().init_run(T)

    def update_action_set(self):
        updated = super().update_action_set()
        if updated:
            self.action_set_proj = self.action_set @ self.phi.T

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
