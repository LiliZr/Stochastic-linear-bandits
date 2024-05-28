import numpy as np

def generate_theta(d, rng):
    theta = rng.randn(d)
    theta /= np.linalg.norm(theta, 2)
    return theta
def generate_action_set(m, d, rng):
    action_set = rng.randn(m, d)
    action_set /= np.linalg.norm(action_set, 2, axis=1)[:, None]
    return action_set