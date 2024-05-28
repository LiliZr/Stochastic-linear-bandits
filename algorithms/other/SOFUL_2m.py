from algorithms.baseline.LinearRegression import *

class LinearRegression_SOFUL_2m(LinearRegression):
    def __init__(self, theta, lam=0.0001, sigma=0.1, delta=0.01, 
                 scale=0.01, action_set=None, seed=48,
                 m=15):
        """
        """
        super().__init__(theta, lam, sigma, delta, scale, action_set, seed)
        self.m = m  
        self.lam = lam
        print(f'- SOFUL 2m... m = {self.m}, {lam}, {scale}')



    def init_run(self, T):
        # Init Variables SOFUL
        self.p = 0
        self.H =  (1/self.lam) * np.identity(self.m)
        self.S = np.zeros((self.m, self.d))
        super().init_run(T, init_Vinv=False)


    def update(self, a_t, r_t):
        ### Update parameters 
        # Update sum reward x action
        self.update_ar(a_t, r_t)
        # Copy old Z
        self.S_t_1 = self.S.copy()
        self.S = np.append(self.S, a_t.reshape((1, -1)), axis=0)

        if (self.S.shape[0] == 2 * self.m):
            # Compute SVD
            _, Sigma, Vt = np.linalg.svd(self.S, full_matrices=False)
            # Get m-th singular value σ² = λ (eigen value) 
            delta = Sigma[self.m-1]**2
            self.p += np.min(Sigma**2)
            # Compute ̂Σ = √ (Σ - δI) (remove useless values that lead to 0 value rows)
            Sigma, Vt = Sigma[:self.m-1], Vt[:self.m-1, :]
            diff = Sigma**2 - delta
            diff[diff < 0] = 0
            S_hat =  np.sqrt(diff)
            # Compute Z
            self.S = (S_hat.reshape(-1, 1) * Vt)
            # Compute H using only the m-th top singular values
            self.H = np.diag(1/(Sigma**2 - delta + self.lam))
        else:
            dim = self.S.shape[0]
            H_tmp = np.zeros((dim, dim))
            p = self.H @ (self.S_t_1 @ a_t)
            k = (a_t @ a_t) - (a_t.T @ (self.S_t_1.T @ p)) + self.lam
            row = - p.T /k 
            column = - p/k
            H_tmp[:dim-1, :dim-1] = self.H + (np.outer(p, p)/k)
            H_tmp[dim-1, :dim-1] = row
            H_tmp[:dim-1, dim-1] = column
            H_tmp[dim-1, dim-1] = 1/k
            self.H = H_tmp

        # Linear regression
        self.theta_est = (1/self.lam) * (self.ar - (self.S.T @ (self.H  @ (self.S @ self.ar))))




############################################################
#                                                          #
#                        SOFUL                             #
#                                                          #
############################################################  

class SOFUL_2m(LinearRegression_SOFUL_2m):
    def recommend(self):
        beta = self.scale * (((self.sigma * np.sqrt((self.m * np.log10(1 + (self.t/(self.m * self.lam))))
                                                   + (2 * np.log10(1/self.delta)) + (self.d * np.log10(1 + (self.p/self.lam))))
                                                   ) * np.sqrt(1 + (self.p / self.lam)))
                             + (np.sqrt(self.lam) * (1 + (self.p / self.lam))))
        


        # Compute UCB for each action
        ucb_max = float('-inf')
        a_max = self.action_set[0]
        self.selected_action_idx = 0
        for idx, a in enumerate(self.action_set):
            Vinv_a =  (1/self.lam) * (a - (self.S.T @ (self.H @ (self.S @ a))))
            ucb = (a @ self.theta_est) + (beta * np.sqrt(a @ Vinv_a))
            if ucb > ucb_max:
                ucb_max = ucb
                a_max = a
                self.selected_action_idx = idx
        return a_max