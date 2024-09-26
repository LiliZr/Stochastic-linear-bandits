#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tracemalloc
import numpy as np
import time

from utils.utils import *



############################################################
#                                                          #
#               Linear Bandit General Class                #
#                                                          #
############################################################

class LinearBandit():
    def __init__(self, theta=None, lam=0.01, sigma=0., delta=0.01, 
                 scale=0.001, action_set=None, seed=48, loop=False):
        """
            theta: True parameter to estimate
            lam: Regularization parameter λ used in regression
            sigma: noise standard deviation σ used in reward computation
            delta: probability param
            scale: scaling factor for beta (ellipsoid param)
            action_set: given action set (implies finite set case)
        """
        self.rng = np.random.RandomState(seed)
        self.theta = theta
        self.action_set = None
        self.d = None
        self.sigma = sigma
        self.lam = lam
        self.scale = scale
        self.delta = delta
        self.finite_set = False
        self.action_set_dict = None
        self.selected_action_idx = -1
        self.rewards = None
        self.L = 1
        self.m2 = 1
        self.max_r, self.min_r = None, None
        self.contextual = False
        self.contextual_arms_rewards = None
        self.loop = loop

        # If an action set is given 
        if action_set is not None:
            self.finite_set = True
            self.action_set = action_set


            if type(action_set) is dict:
                self.action_set_dict = action_set
                self.d = self.action_set_dict[list(self.action_set_dict.keys())[0]].shape[1]
            elif type(action_set) is tuple:
                self.action_set, self.rewards = action_set
                self.d = self.action_set.shape[1]
            elif type(action_set) is list:
                # Choose context randomly
                self.contextual_arms_rewards = action_set
                i = self.rng.randint(0, len(self.contextual_arms_rewards)-1)
                self.action_set = self.contextual_arms_rewards[i][0]
                self.rewards = self.contextual_arms_rewards[i][1]
                self.d = self.action_set.shape[1]
                self.contextual = True
            else:
                self.set_size = len(action_set) 
                self.d = self.action_set.shape[1]





    def generate_reward(self, a_t):
        """
            Generate observed noisy reward given an action
        """
        if self.action_set_dict is not None:
            r = 1 if self.target == self.selected_action_idx else 0
        elif self.rewards is not None:
            # r = self.rewards[self.selected_action_idx] + (self.rng.randn() * self.sigma)
            r = self.rewards[self.selected_action_idx]
        else:
            r_estimated_mean = ((self.theta @ a_t) - self.min_r) / (self.max_r - self.min_r)
            r = r_estimated_mean + (self.rng.randn() * self.sigma)
        return r


    def init_run(self, T):
        """
            Initialize different variables to keep track of reward, CPU time, ...etc
        """
        
        # Init cumulative reward to zero
        self.cumulative_reward = np.empty(T + 1, dtype=object)
        self.cumulative_reward[0], self.cumulative_reward[-1] = 0, None
        # CPU Time
        self.cpu_time = np.zeros(T) 
        # Wall Time
        self.wall_time = np.zeros(T)
        # Memory allocation
        self.memory_peak = np.zeros(T)
        
        # Handle case of multiple classes dataset
        self.sample_action_set = False
        if self.action_set_dict is not None:
            self.nb_classes = len(self.action_set_dict.keys())
            self.target = self.rng.randint(self.nb_classes)
            self.sample_action_set = True
            self.sigma=1
        elif self.action_set is not None and self.theta is not None:
            rewards = self.action_set @ self.theta
            self.min_r = np.min(rewards)
            self.max_r = np.max(rewards)


    def update_action_set(self):
        updated = False
        # Handle case of classification dataset
        if self.sample_action_set:
            self.action_set = []
            for i in range(self.nb_classes):
                idx = self.rng.randint(self.action_set_dict[str(i)].shape[0]-1)
                sample = self.action_set_dict[str(i)][idx]
                self.action_set.append(sample)
            self.action_set = np.array(self.action_set)
            # Normalize
            self.action_set = self.action_set / np.linalg.norm(self.action_set, 2, axis=1)[:, None]
            updated = True
        elif self.contextual:
            # Choose context
            i = self.rng.randint(len(self.contextual_arms_rewards)-1)
            self.action_set, self.rewards = self.contextual_arms_rewards[i]
            updated = True
        return updated


    def update(self, a_t, r_t):
        """
            Update all parameters 
        """
        pass


    def run(self, T=100, time_limit=20):
        # Initialize variables
        self.t0_cpu = time.process_time()
        self.t0_wall = time.perf_counter()

        
        tracemalloc.start()
        tracemalloc.reset_peak()

        self.init_run(T)
        self.t = 0

        while self.t < T:
            # Observe action_set (if changed)
            self.update_action_set()

            # Action recommended 
            a_t = self.recommend()

            # Observed reward
            r_t = self.generate_reward(a_t)

            # Update existing parameters
            self.update(a_t, r_t)

            # Save Times
            self.cpu_time[self.t] =  time.process_time() - self.t0_cpu
            self.wall_time[self.t] =  time.perf_counter() - self.t0_wall

            # Save Memory peak
            _, peak = tracemalloc.get_traced_memory()
            self.memory_peak[self.t] = peak/(1024*1024)
            tracemalloc.reset_peak()
            if self.cpu_time[self.t] > time_limit:
                break
            self.t += 1
            self.cumulative_reward[self.t] =  self.cumulative_reward[self.t - 1] + r_t
        self.memory_peak[self.cpu_time == 0] = None
        self.cpu_time[self.cpu_time == 0] = None
        self.wall_time[self.wall_time == 0] = None
        self.cumulative_reward = np.array(self.cumulative_reward, dtype=float)[1:]
        tracemalloc.stop()

