from matplotlib import pyplot as plt
from algorithms.LinearBandits import *
from algorithms.models import *

from data.load_data import *
from visualizations.plots import *


#### PATH to results 
PATH = './results/main/'
datasets_with_init = ['Movielens', 'Steam', 'Amazon', 'Yahoo']

if __name__ == "__main__":
    ######################## Params ###########################

    # Action set case (List of actions vs L2 Ball)
    finite_set = True
    # Type Dataset in ['MNIST', 'Movielens', 'Steam', 'Amazon', 'Yahoo', 'Random' ]
    dataset = 'Movielens'
    # Max iteration
    T = 500
    # Time limit (in sec)
    time_limit = 40000
    # Number of runs / loops (To approximate in expectation)
    nb_runs = 20
    loop_RP = 5
    seeds = np.arange(nb_runs)
    seed_data = 12

    ######################## Generate theta and action set ###########################
    # Noise
    sigma = 0.1

    df_data = None
    # Load dataset 
    theta, action_set, dfs_data, d, nb_actions = None, None, None, None, None

    print('### Loading data ###')
    if dataset in datasets_with_init:
        dfs_data = load_dataset(name=dataset, init=True)
    else:
        d, nb_actions= 100, 200 # For Random synthetic dataset
        theta, action_set, nb_actions, d = load_dataset(d=d, nb_actions=nb_actions, name=dataset, seed=seed_data)

    ######################## Models ###########################


    #### Other models
    models = [CBSCFD, SOFUL_2m, CBRAP, ConfidenceBall1_FJLT, LinUCB, ConfidenceBall1]
    models_names = [m.__name__ for m in models]
    results = {name:{'reward': np.zeros((nb_runs, T)),
                     'regret': np.zeros((nb_runs, T)),
                     'time_cpu': np.zeros((nb_runs, T)),
                     'time_wall': np.zeros((nb_runs, T)),
                     'memory': np.zeros((nb_runs, T)),}
                                for name in models_names}

    try:
        file_params = "dict_params.json"
        params_dataset = json.load(file_params)
        params = params_dataset[dataset]
        print(params)
    except:
        params =  {'Random':
                {'scale':0, 
                    'lam':0},
                'ConfidenceBall1': 
                    {'scale':0.01, 
                    'lam':0.001},
                'LinUCB': 
                    {'scale':0.003, 
                    'lam':1000},
                'ConfidenceBall1_FJLT': 
                    {'scale':0.1,  
                    'lam':2e-5,
                    'k':50
                    },
                'SOFUL': 
                    {'scale': 1, 
                    'lam': 2e-5,
                    'm':10},
                'SOFUL_2m': 
                    {'scale': 1, 
                    'lam': 0.002,
                    'm':10},
                'CBSCFD': 
                    {'lam': 2e-7,
                    'm':10,
                    'scale':0.1},
                'CBRAP':
                    {'lam':2e-5 ,
                    'k':50,
                    'scale':0.1}
                    } 



    for i in range(nb_runs):
        print(f'____It {i}___')
        if dataset in datasets_with_init:
            # Get new action set of new user at each run
            theta, action_set, nb_actions, d = load_dataset(df_data=dfs_data, name=dataset, seed=seeds[i])
        
        # Optimal model
        optimal = Optimal(theta, action_set=action_set, seed=seeds[i])
        optimal.run(T, time_limit=time_limit)
        reward_max = optimal.cumulative_reward
        # Other models
        for model_, model_name in zip(models, models_names):
            reward_, regret_, time_cpu_, time_wall_, memory_ = [], [], [], [], []
            loop_RP_ = loop_RP if dataset not in ['Random'] and model_.__name__  in ['CBRAP', 'ConfidenceBall1_FJLT',] else 1
            for j in range(loop_RP_):
                if model_name in ['CBRAP', 'ConfidenceBall1_FJLT',]:
                    params[model_.__name__]['seed_proj'] = j
                model = model_(theta, action_set=action_set, sigma=sigma, seed=seeds[i], delta=1/T, **params[model_name])
                model.run(T, time_limit=time_limit)
                reward_.append(model.cumulative_reward)
                regret_.append(reward_max - model.cumulative_reward)
                time_cpu_.append(model.cpu_time)
                time_wall_.append(model.wall_time)
                memory_.append(model.memory_peak)
            results[model_name]['reward'][i] = np.mean(reward_, axis=0)
            results[model_name]['regret'][i] = np.mean(regret_, axis=0)
            results[model_name]['time_cpu'][i] = np.mean(time_cpu_, axis=0)
            results[model_name]['time_wall'][i] = np.mean(time_wall_, axis=0)
            results[model_name]['memory'][i] = np.mean(memory_, axis=0)

    ######################## Plots and results saving ########################
    params_exp = {'d':d, 'T':T, 'nb_actions':nb_actions, 'sigma':sigma, 
                                    'dataset':dataset, 'nb_runs':nb_runs, 'loop_RP': loop_RP}
    path_results = plot_save_intermediate_results(results, params_exp, params_models=params,
                                                    PATH=PATH, params_comparison=False, )