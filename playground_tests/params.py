from itertools import product

from algorithms.LinearBandits import *
from algorithms.select import *
from datasets.load_data import *
from result_functions.plots import plot_regret_time_over_iterations


#### PATH to results 
PATH = './results/parameters/'

if __name__ == "__main__":
    ######################## General Params ###########################
    # Action set case (List of actions vs L2 Ball)
    finite_set = True
    # Type Dataset : 'MNIST' or 'Random'
    dataset = 'MNIST'
    # Time limit (in sec)
    time_stop = 30000
    # Max iterations
    n = 4000
    # Number of loops (To approximate in expectation)
    n_loop = 1
    seeds = np.arange(n_loop)

    ########################  load data  ###########################
    #### Useful if dataset Random
    # Dimension
    d = 2000
    # Number of actions
    m_actions = 100
    # Noise
    sigma = 0.1

    theta, action_set, m_actions, d = load_dataset(d, m_actions, name=dataset, finite_set=finite_set)
    
    ######################## Params for models ###########################
    params_models = {'LinREL1': ['scale', 'lam'],
                    'LinUCB': ['scale', 'lam'],
                    'LinREL1_FJLT': ['c', 'scale', 'lam',  'eps'],
                    'SOFUL': ['m', 'beta', 'lam'],
                    'SOFUL_2m': ['m', 'beta', 'lam'],
                    'CBSCFD': ['m', 'beta', 'lam'],
                    'LRP': ['c0', 'lam', 'm', 'q', 'u', 'omega', 'C_lasso'],
                    'Random':[]
                    }
    # sketched_dim = [30, 35, 40, 45, 55]
    # sketched_dim = [200] # CBSCFD
    sketched_dim = [50] # FJLT

    scale_s = [ 10**(-5) ]
    beta_s = [0.001]
    lam_s = [2]



    params_values = {
          'm':sketched_dim,
          'scale':scale_s,
          'beta':beta_s,
          'lam':lam_s,
          'c': [ m /( np.log10(m_actions)) for m in sketched_dim],
          'eps':[1]
    }
    
    for model_ in [  CBSCFD, LinREL1_FJLT ]:
        params_model = {}
        if model_.__name__ == 'CBSCFD':
            params_values['m'] = [10]
        # if model_.__name__=='LinREL1_FJLT':
        #     params_values['lam'] = [0.0002, 0.002, 0.02, 0.2]
        for p in params_models[model_.__name__ ]:
            params_model[p] = params_values[p]

        for vals in product(*params_model.values()):
            params = dict(zip(params_model, vals))
            params_str = ''.join([f'_{key}={value}' for key, value in params.items()])
            print(model_, params_str)

            #### Models to test
            models = [model_,  ]
            models_names = [m.__name__ for m in models]
            reward = {name: np.zeros((n_loop, n)) for name in models_names}
            regret = {name: np.zeros((n_loop, n)) for name in models_names}
            running_time = {name: np.zeros((n_loop, n)) for name in models_names}


            for i in range(n_loop):
                print(f'____It {i}___')
                #### Optimal model
                optimal = Optimal(theta, action_set=action_set, seed=seeds[i])
                optimal.run(n)
                reward_max = optimal.cumulative_reward
                for model, model_name in zip(models, models_names):
                    model = model(theta, action_set=action_set, sigma=sigma, seed=seeds[i], delta=1/n, **params)
                    model.run(n, time_stop=time_stop)
                    reward[model_name][i] = model.cumulative_reward
                    regret[model_name][i] = reward_max - model.cumulative_reward
                    running_time[model_name][i] = model.time

            params_ = {
                model_.__name__:params
            }
            ######################## Plots ########################
            plot_regret_time_over_iterations(models_names, regret, running_time, reward,
                                            n, d, m_actions, sigma, dataset, params_comparison=True,
                                            PATH=PATH, params=params_)