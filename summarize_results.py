from algorithms.LinearBandits import *
from algorithms.models import *
from data.load_data import *
from visualizations.plots import *


# PATH to results 
PATH = '/users/lahdak/izri/phd/results/parameters/test_cop/'
# PATH = '/users/lahdak/izri/phd/results/Contextual/users=10//'
# PATH = '/users/lahdak/izri/phd/code/users/Stochastic-linear-bandits/results/parameters/'
PATH = '/users/lahdak/izri/phd/results/parameters/06/02/Contextual/users=10/'
PATH = '/users/lahdak/izri/phd/results/parameters/copy/'




# Datasets and models to test
datasets = ['MNIST', 'Movielens', 'Steam', 'Amazon', 'Yahoo' ]
datasets = [ 'Yahoo', 'Steam' ]
datasets = [ 'Movielens']





if __name__ == "__main__":
    ######################## General Params ###########################
    # Time limit (in sec)
    time_limit = 100000
    # Max iterations
    T = 2500 # 3000
    # Number of runs (To approximate in expectation) and seeds
    nb_runs = 50 #20
    loop_RP = 4
    seeds = np.arange(nb_runs)
    seed_data = 12

    dims_dataset = {
        'MNIST': {'k':50, 'm':50},
        'Movielens': {'k':50, 'm':10},
        'Steam': {'k':50, 'm':10},
        'Random': {'k':50, 'm':50},
        'Amazon': {'k':200, 'm':50},
        'Yahoo': {'k':50, 'm':50}
    }
    
    # Original
    dims_dataset = {
        'MNIST': {'k':50, 'm':50},
        'Movielens': {'k':50, 'm':10},
        'Steam': {'k':50, 'm':10},
        'Random': {'k':50, 'm':50},
        'Amazon': {'k':200, 'm':50},
        'Yahoo': {'k':100, 'm':30}
    }


    ########################  load data  ###########################
    # Noise for rewards
    sigma = 0.1

    print('### Loading data ###')
    theta, action_set, dfs_data, d, nb_actions = None, None, None, None, None
    results_mem_all = {} 
    for dataset in datasets:
        # Load dataset 
        if dataset=='Random':
            d, nb_actions= 100, 200 # For Random synthetic dataset

        path_results = path_to_results(dataset, sub_path(d, sigma, nb_actions, T, dataset, nb_runs))
        # plot_ctr(PATH=PATH + path_results, dataset=dataset, **dims_dataset[dataset])
    #     # Plot regret, cpu and wall time over iterations (using best parameters)
        plot_best_results(PATH=PATH + path_results, dataset=dataset, **dims_dataset[dataset])
    #     # Plot regret over parameters scale (ellipsoid) and lambda (regularization)
        # plot_over_params(PATH + path_results, dataset,  **dims_dataset[dataset])
    #     # plot_over_param(PATH + path_results, dataset, param='lam', **dims_dataset[dataset])
    #     # Save memory 
    #     results_mem = results_memory(PATH + path_results, **dims_dataset[dataset])
    #     results_mem_all[dataset] = results_mem
    # # Bar plot memory usage for each dataset
    # plot_memory(results_mem_all, PATH=PATH,)