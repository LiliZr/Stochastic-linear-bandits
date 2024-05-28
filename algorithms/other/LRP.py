from algorithms.baseline.LinearRegression import *

from sklearn import linear_model
from scipy.linalg import block_diag
import cvxpy as cp

class LinearRegression_LRP(LinearBandit):
    def __init__(self, theta, lam=0.0001, sigma=0.1, delta=0.01, 
                 scale=0.01, action_set=None, seed=48,
                 m=15, omega=0.1, u=2, q=15, c0=1, C_lasso=0):
        """
        """
        super().__init__(theta, lam, sigma, delta, scale, action_set, seed)
        self.c0 = c0
        self.c1 = 1./3
        self.m = m  
        self.q = q
        self.u = u
        self.lam0 = lam
        self.omega = omega
        self.C_lasso = C_lasso
        print(f'- LRP... m = {self.m}')




    def init_run(self, n):
        # Init Variables 
        self.epoch = 1
        ## Randomly chosen context vectors 
        self.R = []
        self.y_r = []
        ## All chosen context vectors
        self.W = []
        self.y_w = []
        ## Init θ_0
        self.theta_0 = np.zeros(self.m)
        ## Init D0
        D = np.zeros(self.d)
        indices_D = np.random.choice(self.d, self.q)
        for i in indices_D:
            D[i] = np.sqrt(self.m/self.q)
        self.D0 = np.diag(D)
        ## Init P0
        self.P0 = np.random.normal(0, np.sqrt(1 / self.m), size = (self.m, self.d))  
        ## Init tau
        self.tau = float('+inf')
        ## Init A0 (threshold of significant features)    
        self.A0 = 0
        super().init_run(n, init_Vinv=False)



    def run(self, n=100, time_stop=20):
        self.t0 = time.process_time()
        self.init_run(n)        
        self.t = 0
        self.S_size = 0
        self.S = []
        indices = np.arange(self.d)
        while self.t < n:
            # Handle case of multiple classes dataset
            if self.sample_action_set:
                self.action_set = []
                for i in range(self.nb_classes):
                    idx = self.rng.randint(self.action_set_dict[str(i)].shape[0])
                    sample = self.action_set_dict[str(i)][idx]
                    self.action_set.append(sample)
                self.action_set = np.array(self.action_set)
            #### Beginning of epoch
            if self.t == self.u**self.epoch:
                # Lasso
                self.lam = self.lam0 * np.sqrt((np.log10(self.d) + np.log10(self.t))/len(self.R))
                clf = linear_model.Lasso(alpha=self.lam)
                clf.fit(self.R, self.y_r)
                self.beta_hat = clf.coef_

                # Update S and reconstruct matrices
                ### TODO: are we supposed to know "n" ?
                self.A0 = self.C_lasso * np.sqrt(np.log10(self.d + n)/len(self.R))
                mask = np.abs(self.beta_hat) >= 2 * self.A0
                self.S = indices[mask]
                self.S_size = self.S.shape[0]
                
                ## reconstruct D0
                d_s =  self.d - self.S_size
                D = np.zeros(d_s)
                indices_D = np.random.choice(d_s, self.q)
                for i in indices_D:
                    D[i] = np.sqrt(self.m/self.q)
                D = np.diag(D)
                self.D0 = block_diag(np.identity(self.S_size), D)
                ## reconstruct P0
                P = np.random.normal(0, np.sqrt(1 / self.m), size = (self.m, d_s))
                self.P0 = block_diag(np.identity(self.S_size), P)

                # Permutate features params in b_hat (set significant feats at top)
                for i, j_top_feat in enumerate(self.S):
                    self.beta_hat[i], self.beta_hat[j_top_feat] = self.beta_hat[j_top_feat], self.beta_hat[i]

                # Update epoch and theta_0
                self.epoch += 1
                self.theta_0 = self.P0 @ self.beta_hat
                self.tau = self.A0

            # Draw random var
            t = self.t
            t+=1
            P_c0 = min(1, self.c0 * (t)**(-self.c1))
            r_t = np.random.binomial(1, P_c0)
            y_t = 0
            ## Random sample
            if r_t == 1:
                self.selected_action_idx = np.random.randint(0, self.action_set.shape[0])
                a_t = self.action_set[self.selected_action_idx]
                self.R.append(a_t)
                y_t = self.generate_reward(a_t)
                self.y_r.append(y_t)
            ## UCB
            else:
                dim = self.S_size + self.m
                # Permutate features of context_vectors (set significant feats at top) and project them
                proj_W = np.array(self.W)
                for i, j_top_feat in enumerate(self.S):
                    proj_W[:, i], proj_W[:, j_top_feat] = proj_W[:, j_top_feat], proj_W[:, i]
                proj_W = (self.P0 @ self.D0 @ proj_W.T).T
                # Solve linear regression with constraint (compute ̂θ  with ||θ - θ_0|| ≤ τ) 
                theta = cp.Variable(dim)
                objective = cp.Minimize(cp.sum_squares(proj_W @ theta - self.y_w))
                constraints = [cp.norm2(theta - self.theta_0) <= self.tau]
                prob = cp.Problem(objective, constraints)
                result = prob.solve(solver=cp.CLARABEL)
                self.theta_est = theta.value
                # Inverse cov matrix
                self.Winv = np.linalg.pinv(proj_W.T @ proj_W)
                # Action recommended 
                a_t = self.recommend()
                # Observed reward
                y_t = self.generate_reward(a_t)
            self.W.append(a_t)
            self.y_w.append(y_t)


            self.cpu_time[self.t] =  time.process_time() - self.t0
            if self.cpu_time[self.t] > time_stop:
                break
            self.t += 1
            self.cumulative_reward[self.t] = self.cumulative_reward[self.t - 1] + y_t
        self.cpu_time[self.cpu_time == 0] = None
        self.cumulative_reward = np.array(self.cumulative_reward, dtype=float)[1:]
        print(f'_____________{self.S_size + self.m}_____________')



############################################################
#                                                          #
#                          LRP                             #
#                                                          #
############################################################  

class LRP(LinearRegression_LRP):
    def recommend(self):
        # omega = ((self.S_size + self.m) * np.log10(self.t)**0.5) + (((self.S_size + self.m)**0.25 * np.sqrt(self.t * A1 * self.tau * np.log10(self.t))))
        # Compute UCB for each action
        ucb_max = float('-inf')
        a_max = self.action_set[0]
        self.selected_action_idx = 0
        # Permutate features
        action_set_ = self.action_set.copy()
        for i, j_top_feat in enumerate(self.S):
            action_set_[:, i], action_set_[:, j_top_feat] = action_set_[:, j_top_feat], action_set_[:, i]

        # Projection
        self.action_set_proj = (self.P0 @ self.D0 @ action_set_.T).T

        for idx, a in enumerate(self.action_set_proj):
            ucb = (a @ self.theta_est) + (self.omega * np.sqrt(a @ (self.Winv @ a)))
            if ucb > ucb_max:
                ucb_max = ucb
                a_max = a
                self.selected_action_idx = idx
        return  self.action_set[self.selected_action_idx]