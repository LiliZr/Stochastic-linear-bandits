
import tracemalloc

from sklearn.decomposition import PCA
from algorithms.baseline.LinearRegression import *


class LinearRegression_PCA(LinearRegression):
    def __init__(self, theta, lam=0.0001, sigma=0.1, delta=0.01, 
                 scale=0.01, action_set=None, seed=48,
                 m=10, nb_samples=1000):
        """
            c: constant used in matrix embedding generation
            eps: distortion parameter in FJLT
            p: kind of embedding we project to
            m: number of samples from L2 ball in infinite case
        """
        LinearRegression.__init__(self, theta, lam, sigma, delta, scale, action_set, seed)
        self.nb_samples = nb_samples # Samples L2 Ball
        self.m = m       


    def init_run(self, n):
        self.pca = PCA(n_components=self.m)
        self.d = self.m
        # L2 ball
        if self.action_set is None:
            self.action_set = sample_uniform_d_sphere(self.nb_samples, self.theta.shape[0]-1)

        # Project action set in embedding space
        if self.action_set_dict is None:
            self.action_set_proj = self.pca.fit_transform(self.action_set)
        
        # Initialization of general structures
        super().init_run(n)

        # Upper bound for action set
        if self.action_set_dict is None:
            if self.action_set is not None:
                self.L = max(np.linalg.norm(self.action_set_proj, 2, axis=1))

            if self.theta is not None:
                # Upper bound for theta
                self.m2 = np.linalg.norm(self.theta, 2)

    def update_action_set(self):
        updated = super().update_action_set()
        if updated:
            self.action_set_proj = self.pca.fit_transform(self.action_set)

    def update(self, a_t, r_t):
        # Update parameters
        a_proj_t, a_t = a_t
        self.update_ar(a_proj_t, r_t)

        # Linear regression
        inv_sherman_morrison(self.Vinv, a_proj_t)
        self.theta_est = self.Vinv @ self.ar 
        



class LinREL1_PCA(LinearRegression_PCA):
    """ 
        LinREL1_PCA class recommending over a finite set
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


