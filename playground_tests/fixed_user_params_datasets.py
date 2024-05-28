from itertools import product

from algorithms.LinearBandits import *
from algorithms.models import *
from datasets.load_data import *
from visualizations.plots import *


# PATH to results 
PATH = './results/parameters/fixed_user_5/'

# Datasets and models to test
datasets = ['Steam']

models_to_test = [ConfidenceBall1_FJLT]

# Hyperparameters of each model
params_models = {'ConfidenceBall1': ['scale', 'lam'],
                    'LinUCB': ['scale', 'lam'],
                    'ConfidenceBall1_FJLT': ['k', 'scale', 'lam'],
                    'ConfidenceBall1_FJLT_Cholesky': ['k', 'scale', 'lam'],
                    'SOFUL': ['m', 'beta', 'lam'],
                    'SOFUL_2m': ['m', 'scale', 'lam'],
                    'CBSCFD': ['m', 'scale', 'lam'],
                    'CBRAP':['lam', 'scale', 'k'],
                    'Random':[]}

sketch_dim = [10,]
projected_dim = [50]

scale_s = [10**i for i in range(-2, 0)]
lam_s = [2*10**i for i in range(-7, 0, 2)]

scale_s = [0.1]
# lam_s = [2*10**i for i in range(-7, 0, 2)]
# lam_s = [2e-5]


seed_user = 14

# scale_s = [0.1]
# lam_s = [2e-9]



params_values = {
    'm':sketch_dim,
    'scale':scale_s,
    'lam':lam_s,
    'k':projected_dim,
}


if __name__ == "__main__":
    ######################## General Params ###########################
    # Time limit (in sec)
    time_limit = 30000
    # Max iterations
    T = 2500
    # Number of runs (To approximate in expectation) and seeds
    nb_runs = 50
    loop_RP = 4
    seeds = np.arange(nb_runs)
    seed_data = 12

    ########################  load data  ###########################
    # Noise for rewards
    sigma = 0.1

    print('### Loading data ###')
    theta, action_set, dfs_data, d, nb_actions = None, None, None, None, None
    for dataset in datasets:
        # Load dataset 
        if dataset in ['Movielens', 'Steam']:
            dfs_data = load_dataset(name=dataset, init=True)
        else:
            d, nb_actions= 100, 200 # For Random synthetic dataset
            theta, action_set, nb_actions, d = load_dataset(d=d, nb_actions=nb_actions, name=dataset, seed=seed_data)

        nb_runs_ = None
        loop_RP_ = None
        for model_ in models_to_test:
                print('### Testing models ###')
                # Set number of runs according to model and dataset 
                nb_runs_ = nb_runs if model_.__name__  not in ['LinUCB', 'ConfidenceBall1'] else 1

                # nb_runs_ = nb_runs if model_.__name__  not in [] else 1
                loop_RP_ = loop_RP if dataset not in ['Random'] and model_.__name__  in ['CBRAP', 'ConfidenceBall1_FJLT', 'ConfidenceBall1_FJLT_Cholesky'] else 1
                # Get hyperparameters and combination of values to test
                params_model = {}
                for p in params_models[model_.__name__ ]:
                    params_model[p] = params_values[p]
                for vals in product(*params_model.values()):
                    try:
                        params = dict(zip(params_model, vals))
                        params_str = ''.join([f'_{key}={value}' for key, value in params.items()])
                        print(model_, params_str)
                        #### Models to test
                        models = [model_,  ]
                        models_names = [m.__name__ for m in models]
                        results = {name: 
                                {'reward': np.zeros((nb_runs_, T)),
                                    'regret': np.zeros((nb_runs_, T)),
                                    'time_cpu': np.zeros((nb_runs_, T)),
                                    'time_wall': np.zeros((nb_runs_, T)),
                                    'memory': np.zeros((nb_runs_, T)),}
                                for name in models_names}


                        for i in range(nb_runs_):
                            print(f'{dataset}____It {i}___{params_str}')
                            if dataset in ['Movielens', 'Steam']:
                                # Get new action set of new user at each run
                                theta, action_set, nb_actions, d = load_dataset(df_data=dfs_data, name=dataset, seed=seed_user)
                            
                            # Optimal model
                            optimal = Optimal(theta, action_set=action_set, seed=seeds[i])
                            optimal.run(T, time_limit=time_limit)
                            reward_max = optimal.cumulative_reward
                            reward_, regret_, time_cpu_, time_wall_, memory_ = [], [], [], [], []
                            for j in range(loop_RP_):
                                print(f'loop: {j}')
                                if model_.__name__ in ['CBRAP', 'ConfidenceBall1_FJLT', 'ConfidenceBall1_FJLT_Cholesky']:
                                    params['seed_proj'] = j
                                # Other models
                                for model, model_name in zip(models, models_names):
                                    model = model(theta, action_set=action_set, sigma=sigma, seed=seeds[i], delta=1/T, **params)
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
                        params_ = {
                            model_.__name__:params
                        }
                        params_exp = {'d':d, 'T':T, 'nb_actions':nb_actions, 'sigma':sigma, 
                                    'dataset':dataset, 'nb_runs':nb_runs, 'loop_RP': loop_RP}

                        plot_regret_time_over_iterations(results, params_exp, params_models=params_,
                                                                        PATH=PATH, params_comparison=True, )
                    except Exception as e:
                        with open(f"logs/log_{dataset}_{model_.__name__}.txt", "a") as f:
                            f.write(dataset + '\n' + model_.__name__ + '\n' + params_str + '\n')
                            f.write(str(e))
                            f.write('#############\n')
                        print(dataset + '\n' + model_.__name__ + '\n' + params_str + '\n')
                        print(e)
                        pass
        path_results = path_to_results(dataset, sub_path(d, sigma, nb_actions, T, dataset, nb_runs))
        plot_best_results(PATH=PATH + path_results, dataset=dataset)
