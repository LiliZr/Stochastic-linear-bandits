from algorithms.fjlt.LinearRegression_FJLT import *

############################################################
#                                                          #
#                    ThompsonSampling                      #
#                                                          #
############################################################  

class ThompsonSampling_FJLT(LinearRegression_FJLT):
    def update_V(self, Vinv, a_t):
        """
            Update V with given action for thompson sampling
        """
        self.V = inv(Vinv + (np.outer(a_t, a_t)/(self.sigma**2)))


    def update_ar(self, a_t, r_t):
        """
            Compute reward x action for thompson sampling
            a_t * (r_t + w) / σ² 
        """
        w = np.random.normal(0, self.sigma)
        r_t = (r_t + w) / (self.sigma**2)
        self.ar =  (a_t * r_t)


    def run(self, n=100, time_stop=20):
        # Initialize variables
        self.init_run(n)
        self.theta_est = np.random.normal(size=self.d)
        self.t0 = time.perf_counter()
        self.t = 0

        while self.t < n:
            # Action recommended 
            a_proj_t, a_t = self.recommend()
    
            # Observed reward
            r_t = self.generate_reward(a_t)

            # Update parameters
            Vinv = inv(self.V)
            self.update_V(Vinv, a_proj_t)
            self.update_ar(a_proj_t, r_t)

            # Update estimator
            self.theta_est = self.V @ ((Vinv @ self.theta_est) + self.ar)

            self.cpu_time[self.t] =  time.perf_counter() - self.t0
            self.t += 1
            if self.cpu_time[self.t] > time_stop:
                break
            self.cumulative_reward[self.t] =  self.cumulative_reward[self.t - 1] + r_t
        self.cpu_time[self.cpu_time == 0] = None
        self.cumulative_reward[self.cumulative_reward == 0] = None


    def recommend(self):
        """
            Recommend over finite or infite action set
        """
        # Get the action with highest value
        best_action_idx = np.argmax(self.action_set_proj @ self.theta_est)
        return self.action_set_proj[best_action_idx], self.action_set[best_action_idx]