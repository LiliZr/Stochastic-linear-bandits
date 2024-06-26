import tracemalloc
from algorithms.baseline.LinearRegression import *
import scipy

class LinearRegression_CBSCFD(LinearRegression):
    def __init__(self, theta, lam=0.0001, sigma=0.1, delta=0.01, 
                 scale=0.01, action_set=None, seed=48,
                 m=15):
        """
        """
        super().__init__(theta, lam, sigma, delta, scale, action_set, seed)
        self.m = m  
        self.lam = lam
        print(f'- CBSCFD... m = {self.m}, {lam}, {scale}')



    def init_run(self, n):
        # Init Variables CBSCFD
        self.alpha = self.lam
        self.Z = np.zeros((self.m, self.d))
        self.H = (1/self.alpha) * np.identity(self.m)

        super().init_run(n, init_Vinv=False)

    def update(self, a_t, r_t):
         ### Update parameters 
        # Update sum reward x action
        self.update_ar(a_t, r_t)
        # Copy old Z
        self.Z_t_1 = self.Z.copy()
        self.Z = np.append(self.Z, a_t.reshape((1, -1)), axis=0)

        if (self.Z.shape[0] == 2 * self.m):
            # Compute SVD
            _, S, Vt = np.linalg.svd(self.Z, full_matrices=False)
            # Get m-th singular value²
            delta = S[self.m-1]**2
            # Add this value to alpha
            self.alpha += delta
            # Compute ̂Σ = √ (Σ² - δI) (remove empty rows (0 value rows))
            S, Vt = S[:self.m-1], Vt[:self.m-1, :]
            diff = S**2 - delta
            S_hat =  np.sqrt(diff)
            # Compute Z
            self.Z = (S_hat.reshape(-1, 1) * Vt)
            # Compute H using only the m-th top singular values
            self.H = np.diag(1/(S**2 - delta + self.alpha))
        else:
            dim = self.Z.shape[0]
            H_tmp = np.zeros((dim, dim))
            p = self.H @ (self.Z_t_1 @ a_t)
            k = (a_t @ a_t) - (a_t.T @ (self.Z_t_1.T @ p)) + self.alpha
            row = - p.T /k 
            column = - p/k
            H_tmp[:dim-1, :dim-1] = self.H + (np.outer(p, p)/k)
            H_tmp[dim-1, :dim-1] = row
            H_tmp[:dim-1, dim-1] = column
            H_tmp[dim-1, dim-1] = 1/k
            self.H = H_tmp

        # Linear regression
        self.theta_est = (1/self.alpha) * (self.ar - (self.Z.T @ (self.H  @ (self.Z @ self.ar))))




############################################################
#                                                          #
#                       CBSCFD                             #
#                                                          #
############################################################  

class CBSCFD(LinearRegression_CBSCFD):
    def recommend(self):
        # Compute UCB for each action
        beta = self.scale * ((self.sigma * np.sqrt((2 * np.log10(1/self.delta))
                                     + (self.m * np.log10(1 + (self.t / (self.m * self.lam))))
                                     + (self.d * np.log10(1 + ((self.alpha - self.lam) / self.lam)))))
                + np.sqrt(self.alpha))
            
        ucb_max = float('-inf')
        a_max = self.action_set[0]
        self.selected_action_idx = 0
        HZa = np.zeros(self.H.shape[0])
        ZtHZa = np.zeros(self.Z.shape[1])
        for idx, a in enumerate(self.action_set):
            np.matmul(self.Z, a, out=HZa)
            np.matmul(self.H, HZa, out=HZa)
            np.matmul(self.Z.T, HZa, out=ZtHZa)
            Vinv_a =  (1/self.alpha) * (a - ZtHZa)
            ucb = (a @ self.theta_est) + (beta * np.sqrt(a @ Vinv_a))
            if ucb > ucb_max:
                ucb_max = ucb
                a_max = a
                self.selected_action_idx = idx
        return a_max