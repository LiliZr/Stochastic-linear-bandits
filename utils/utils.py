import numpy as np
import math
import random as rnd
import scipy


from scipy.linalg import hadamard


############################################################
#                                                          #
#                     General functions                    #
#                                                          #
############################################################
def inv_sherman_morrison(B, u):
    """
        Efficient Inverse of 1-rank update 
            return : (B + uu.T)^-1
    """
    u = u.reshape(-1, 1)
    Bu = B @ u
    np.subtract(B, (Bu @ (u.T @ B)) / (1 + (u.T @ Bu)), out=B)


def extrema(Binv, c, cholesky=False):
    """
        Return the extreme points of set:
           x :  ‖B^{1/2} x‖_1 ≤ c
    """
    sqrtBinv = None
    # Not sure about this, it works experimentally but I can't see why 
    if cholesky:
        sqrtBinv = np.linalg.cholesky(Binv)
    else:
        rho_s, U = None, None
        try:
            rho_s, U = np.linalg.eigh(Binv, UPLO='L')
        except:
            rho_s, U = np.linalg.eigh(Binv, UPLO='U')
        sqrtBinv = (U * np.sqrt(rho_s)) @ U.T

    n, n = sqrtBinv.shape
    basis = np.eye(n)
    nbasis = -basis
    pnbasis = np.concatenate((basis, nbasis))
    return c * pnbasis @ sqrtBinv.T

def project_L2_Ball(theta, r=1):
    """
        Return Projection in L2 ball of given theta and its norm.
    """
    norm = np.linalg.norm(theta, 2)
    return (r * (theta / norm), norm)


def regret_bound(time_steps, d, lam=1, scale=1, sigma=1, fjlt=False, eps=0.1): 
    """ 
        Return theoretical regret bound
            R_n ≤ √ (8d²nβlog(1 + n/dλ))
            source: https://tor-lattimore.com/downloads/book/book.pdf
    """
    n = len(time_steps)
    delta=1/n
    beta_sqrt = ((np.sqrt(lam) ) + (sigma * ((np.sqrt((- 2 * np.log10(delta))) + 
                    (d * np.log10(1 + ((time_steps )/(lam*d))))) )) ) * scale 
    regret = 0
    if not fjlt:
        regret = beta_sqrt * np.sqrt(8 * d**2 * time_steps * np.log10(1 + (time_steps/(d*lam)))) 
    else:
        # regret = 2 * np.sqrt(time_steps 
        #                      + (2 * time_steps * (d**2) * (beta_sqrt**2) * np.log10(1 + ((time_steps )/(lam*d)))) 
        #                      + (2 * (time_steps**(3/2)) * np.sqrt(d * beta_sqrt**2)))
        regret = 2 * np.sqrt(((time_steps**2) * (eps**2)) 
                             + (2 * time_steps * (d**2) * (beta_sqrt**2) * np.log10(1 + ((time_steps )/(lam*d)))) 
                             + (2 * (time_steps**2) * eps *  np.sqrt(d * beta_sqrt**2)))
    return regret



def sample_uniform_d_sphere(n, d):
    """
        Sample uniformly from d-sphere
            x: ‖x‖_2 = 1
            source: http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
        n: number of samples
        d: dimension  

    """
    samples = np.random.normal(0, 1, size=(n, d+1))
    norms = np.linalg.norm(samples, 2, axis=1)
    return samples/norms[:,None]


def sample_uniform_d_ball(n, d):
    """
        Sample uniformly from d-ball
            ‖x‖_2 ≤ 1
            source: http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
        n: number of samples
        d: dimension
    """
    samples = sample_uniform_d_sphere(n, d-1)
    c = np.random.uniform(0, 1, size=n)
    return (c ** (1/d))[:,None] * samples


############################################################
#                                                          #
#               Space Embedding FJLT function              #
#                                                          #
############################################################


def generate_phi(d, n, c=0.001, k=None, eps=0.1, p=2, c1=1, seed=None):
    """
        Generate FJLT matrix
            source: https://www.cs.princeton.edu/~chazelle/pubs/FJLT-sicomp09.pdf
        n: number of points
        d: dimension
        k: projected dimension
        c: constant used in matrix embedding generation
        eps: distortion parameter in FJLT
        p: kind of embedding we project to

    """
    rng = np.random.RandomState(seed)
    if k is None:
        k = math.ceil(c * eps**(-2) * np.log10(n))
    # In case the original dim is not a power 2 
    power = np.log2(d)
    next_dim_power_2 = d if power == int(power) else 2**(int(power)+1)

    ## Generate P
    q = min(1, (c1 * (eps**(p-2) ) * (np.log10(n)**p))/next_dim_power_2)
    P = np.zeros((k, next_dim_power_2))
    probabilities = rng.random((k, next_dim_power_2))
    n_samples = len(probabilities[probabilities <= q])
    P[probabilities <= q] = rng.normal(0, np.sqrt(1 / q), size = n_samples)

    ## Generate H
    H = hadamard(next_dim_power_2)[:, :d] * (1./np.sqrt(next_dim_power_2))

    ## Generate D
    diag = rng.choice([-1, 1], p=[0.5, 0.5], size=d)

    return (1 /np.sqrt(k)) * P @ (H * diag)



def idx_nearest_vector(v, v_set):
    distances = np.linalg.norm(v_set - v, 2, axis=1)
    return np.argmin(distances)