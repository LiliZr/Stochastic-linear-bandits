from itertools import product
from tqdm import tqdm

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
    dataset = 'Random'
    # Time limit (in sec)
    time_stop = 30000
    # Max iterations
    n = 1000
    # Number of loops (To approximate in expectation)
    n_loop = 20
    seeds = np.arange(n_loop)

    ########################  load data  ###########################
    #### Useful if dataset Random
    if dataset == 'Random':
        seed=12
        PATH = PATH + f'seed_data={seed}/'
    # Dimension
    d = 1000
    # Number of actions
    m_actions = 1000
    # Noise
    sigma = 0.3

    for m_actions in [10000, 100]:
        for sigma in [0.2]:
            for d in [1000, 5000]:
                print(f'd={d}_actions={m_actions}')
                # if d==1000 and m_actions==100:
                #     continue
                theta, action_set, m_actions, d = load_dataset(d, m_actions, name=dataset, finite_set=finite_set, seed=seed)
                
                ######################## Params for models ###########################
                params_models = {'LinREL1': ['scale', 'lam'],
                                'LinUCB': ['scale', 'lam'],
                                'LinREL1_FJLT': ['c', 'scale', 'lam', 'eps'],
                                'SOFUL': ['m', 'beta', 'lam'],
                                'SOFUL_2m': ['m', 'beta', 'lam'],
                                'CBSCFD': ['m', 'beta', 'lam'],
                                'LRP': ['c0', 'lam', 'm', 'q', 'u', 'omega', 'C_lasso'],
                                'LinREL1_PCA':['lam', 'scale', 'm'],
                                'CBRAP':['lam', 'beta', 'm'],
                                'Random':[]
                                }
                sketched_dim = [50]
                scale_s = [(10**i) for i in range(-4, 0)]
                scale_s = [0.01]
                beta_s = [10**i for i in range(-4, 2)]#0.001/0.2, 0.1/20
                # beta_s = [0.01 ]#0.001/0.2, 0.1/20
                lam_s = [2*(10**i) for i in range(-4, 2)]
                lam_s = [2]


                params_values = {
                    'm':sketched_dim,
                    'scale':scale_s,
                    'beta':beta_s,
                    'lam':lam_s,
                    'c': [ m /( np.log10(m_actions)) for m in sketched_dim],
                    'eps':[1]
                }
                
                for model_ in [  LinREL1_FJLT]:
                    params_model = {}
                    if model_.__name__=='CBRAP':
                        params_values['beta'] = scale_s
                    for p in params_models[model_.__name__ ]:
                        params_model[p] = params_values[p]
                    # params_model =best_params[d][m_actions][model_.__name__]

                    for vals in tqdm(product(*params_model.values())):
                        try:
                            params = dict(zip(params_model, vals))
                            params_str = ''.join([f'_{key}={value}' for key, value in params.items()])
                            print(model_, params_str)

                            #### Models to test
                            models = [model_,  ]
                            models_names = [m.__name__ for m in models]
                            reward = {name: np.zeros((n_loop, n)) for name in models_names}
                            regret = {name: np.zeros((n_loop, n)) for name in models_names}
                            time_cpu = {name: np.zeros((n_loop, n)) for name in models_names}
                            memory = {name: np.zeros((n_loop, n)) for name in models_names}



                            for i in range(n_loop):
                                print(f'____It {i}___')
                                #### Optimal model
                                optimal = Optimal(theta, action_set=action_set, seed=seeds[i])
                                optimal.run(n, time_stop=time_stop)
                                reward_max = optimal.cumulative_reward
                                for model, model_name in zip(models, models_names):
                                    model = model(theta, action_set=action_set, sigma=sigma, seed=seeds[i], delta=1/n, **params)

                                    # mem = memory_usage((model.run, (n, time_stop)),  include_children=True)
                                    # print(len(mem), np.max(mem))

                                    model.run(n, time_stop=time_stop)
                                    reward[model_name][i] = model.cumulative_reward
                                    regret[model_name][i] = reward_max - model.cumulative_reward

                                    time_cpu[model_name][i] = model.time
                                    memory[model_name][i] = model.memory_peak

                            params_ = {
                                model_.__name__:params
                            }
                            ######################## Plots ########################
                            plot_regret_time_over_iterations(models_names, regret, time_cpu, reward, memory,
                                                            n, d, m_actions, sigma, dataset, params_comparison=True,
                                                            PATH=PATH, params=params_)
                        except:
                            pass