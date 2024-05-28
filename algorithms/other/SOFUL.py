from algorithms.baseline.LinearRegression import *

class LinearRegression_SOFUL(LinearRegression):
    def __init__(self, theta, lam=0.0001, sigma=0.1, delta=0.01, 
                 scale=0.01, action_set=None, seed=48,
                 m=15, beta=0.1):
        """
        """
        super().__init__(theta, lam, sigma, delta, scale, action_set, seed)
        self.m = m  
        self.lam = lam
        self.beta = beta
        print(f'- SOFUL... m = {self.m}')


    def fd_sketching(self, S, a, lam=0.1):
        l, d = S.shape 
        s = (S.T @ S) + np.outer(a, a)
        rho_s, U = np.linalg.eigh(s)

        # Order eigen vectors and value  
        rho_s = rho_s[::-1]
        U = U[:, ::-1]

        # Keep m first eigen vectors
        rho_s, U = rho_s[:l], U[:, :l]

        S = (np.sqrt(rho_s - rho_s[-1])).reshape(-1, 1) * U.T
        H = np.diag(1/((rho_s - rho_s[-1]) + lam))
        return S, H


    def init_run(self, T):
        # Init Variables SOFUL
        self.S = np.zeros((self.m, self.d))
        self.H =  (1/self.lam) * np.identity(self.m)
        super().init_run(T, init_Vinv=False)



    def update(self, a_t, r_t):
        # Update parameters
        self.update_ar(a_t, r_t)
        self.S, self.H = self.fd_sketching(self.S, a_t, self.lam)

        # Linear regression
        self.theta_est = (1/self.lam) * (self.ar - (self.S.T @ (self.H  @ (self.S @ self.ar))))


############################################################
#                                                          #
#                        SOFUL                             #
#                                                          #
############################################################  

class SOFUL(LinearRegression_SOFUL):
    def recommend(self):
        # Compute UCB for each action
        ucb_max = float('-inf')
        a_max = self.action_set[0]
        self.selected_action_idx = 0
        for idx, a in enumerate(self.action_set):
            Vinv_a =  (1/self.lam) * (a - (self.S.T @ (self.H @ (self.S @ a))))
            ucb = (a @ self.theta_est) + (self.beta * np.sqrt(a @ Vinv_a))
            if ucb > ucb_max:
                ucb_max = ucb
                a_max = a
                self.selected_action_idx = idx
        return a_max