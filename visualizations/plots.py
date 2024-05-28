import math
import numpy as np
import json
import os
import glob as gb
import re
import matplotlib.pyplot as plt

from functools import reduce

PATH = ''
numbers = re.compile(r'(\d+)')

#plt.style.use('tableau-colorblind10')

projected_dim = 50
sketch_dim = 50
sketch_dim_sof = 50
dim_model = {'ConfidenceBall1_FJLT': projected_dim,
            'CBSCFD': sketch_dim, 
            'CBRAP': projected_dim,
            'ConfidenceBall1_PCA': projected_dim,
            'CBRAP_lam1_sig1': projected_dim,
            'CBRAP_sig1':projected_dim,
            'ConfidenceBall1_FJLT_Cholesky': projected_dim,
            'SOFUL_2m': sketch_dim_sof}

colors =   {'ConfidenceBall1_FJLT': 'dodgerblue',
            'CBSCFD': 'mediumaquamarine', 
            'CBRAP': 'darkorange',
            'SOFUL_2m': 'hotpink',
            'ConfidenceBall1': 'blue',
            'LinUCB':'darkorchid',
            'Random': 'crimson'}

spacing =   {'ConfidenceBall1_FJLT': 1,
            'CBSCFD': 1, 
            'CBRAP': 1.5,
            'SOFUL_2m': 1.5,
            'ConfidenceBall1': 1,
            'LinUCB':1.5,
            'Random': 1}

markers = {'ConfidenceBall1_FJLT': 'x',
            'CBSCFD': 'p', 
            'CBRAP': 's',
            'SOFUL_2m': '^',
            'ConfidenceBall1': '*',
            'LinUCB':'o',
            'Random': 'd'}


hatchs = {'ConfidenceBall1_FJLT': '//',
            'CBSCFD': '\\\\', 
            'CBRAP': '//',
            'SOFUL_2m': '\\\\',
            'ConfidenceBall1': 'xx',
            'LinUCB':'xx',
            'Random':'..'}

list_models_to_not_plot_s = {'MNIST': ['Random',],
                           'Random': ['Random', 'LinUCB', 'ConfidenceBall1'],
                           'Steam': ['Random', 'LinUCB', 'ConfidenceBall1','' ],
                           'Movielens': ['Random'   ],
                           'Amazon': ['Random'],
                           'Yahoo': ['Random']}

def numerical_sort(value):
    """
        Order numerical values 
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def intersect(lst1, lst2):
    """
        Intersect two lists
    """
    return list(set(lst1).intersection(lst2))


def sub_path(d, sigma, nb_actions, T, dataset, nb_runs):
    """
        Create subpath to results given dataset and parameters
    """
    sub_PATH = f'nb_runs={nb_runs}/T={T}/'
    sub_PATH += f'/d={d}_σ={sigma}_actions={nb_actions}/' if dataset == 'Random' else ''
    return sub_PATH


def path_to_results(dataset, sub_PATH):
    """
        Create path to results given dataset 
    """
    return f"{PATH}/{dataset}/{sub_PATH}/"


def define_label_title(model_name, params, 
                       d, n, nb_actions, dataset, nb_runs,
                       params_comparison, regret='', sigma=None):
    """
        Create corresponding label and subtitle given model and params
    """
    label = f'{model_name}' 
    title = f'{regret}_{dataset}' if params_comparison else dataset
    title += f'_n={n}_d={d}' 
    title += f'_σ={sigma}_actions={nb_actions}' if dataset == 'Random' else ''
    sub_PATH = sub_path(d, sigma, nb_actions, n, dataset,  nb_runs)
    sub_PATH_model = sub_PATH
    sub_PATH_model += f'/{model_name}/' if params_comparison else ''


    if params is not None:
        if 'Random' != model_name:
            for key, value in params.items():
                if key == 'm':
                    label += f'_{key}={value}'
                    title += f'_{key}={value}_' if params_comparison else ''
                    sub_PATH_model += f'/{value}/' if params_comparison else ''
                elif key == 'c' or key == 'k':
                    k = None
                    if key == 'c':
                        k = math.ceil(value * params['eps']**(-2) * np.log10(nb_actions))
                    else:
                        k = value
                    label += f'_k={k}'
                    title += f'_k={k}_' if params_comparison else ''
                    sub_PATH_model += f'/{k}/' if params_comparison else ''

                elif key != 'eps':
                    title += f'_{key}={value}_' if params_comparison else ''
            if params_comparison:
                title = title[:-1]

    return label, title, sub_PATH_model, sub_PATH
   

def plot_save_intermediate_results(results, params_exp, params_models,
                                     PATH='./', params_comparison=False):
    """
        Plot intermediate results (regret, time, memory) over iterations and save them as json
    """
    d, T, nb_actions, sigma, dataset = params_exp['d'], params_exp['T'], params_exp['nb_actions'], params_exp['sigma'], params_exp['dataset']
    models_names = results.keys()
    sub_PATH_model = ''
    sub_PATH = ''
    list_models_to_not_plot = list_models_to_not_plot_s[dataset]


    ######################## Plots ########################
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))

    ##### json saving
    results_json = {}
    regret_final = 0

    ######## Plot models
    time_steps = np.arange(T)

    for model_name in models_names:
        # Regret
        regret_mean = np.mean(results[model_name]['regret'], axis=0)
        regret_final = np.round(regret_mean[-1], decimals = 3)
        regret_std = np.std(results[model_name]['regret'], axis=0)

        # Define label and title
        label, title, sub_PATH_model, sub_PATH = define_label_title(model_name, params_models[model_name], 
                                                                    d, T, nb_actions, dataset, params_exp['nb_runs'],
                                                                    params_comparison, str(regret_final), sigma)
        ax1.plot(time_steps, regret_mean, label=label, color=colors[model_name], marker=markers[model_name], markersize=5, markevery=int(spacing[model_name]*T/10))
        ax1.fill_between(time_steps, regret_mean - regret_std, regret_mean + regret_std, alpha=0.2, color=colors[model_name])

        # Reward
        reward_mean = np.mean(results[model_name]['reward'], axis=0)
        ax2.plot(time_steps, reward_mean, label=label, color=colors[model_name], marker=markers[model_name], markersize=5, markevery=int(spacing[model_name]*T/10))
        reward_std = np.std(results[model_name]['reward'], axis=0)
        ax2.fill_between(time_steps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, color=colors[model_name],)

        # CPU time
        time_cpu_mean = np.mean(results[model_name]['time_cpu'], axis=0)
        ax3.scatter(time_steps[-1], np.round(time_cpu_mean[-1], decimals = 3), label=np.round(time_cpu_mean[-1]), marker='P', color=colors[model_name],)
        ax3.plot(time_steps, time_cpu_mean, label=model_name, color=colors[model_name], marker=markers[model_name], markersize=5, markevery=int(spacing[model_name]*T/10))

        # Wall Time
        time_wall_mean = np.mean(results[model_name]['time_wall'], axis=0)

        # Memory
        memory_max = np.around(np.max(results[model_name]['memory'], axis=0), decimals=3)
        ax4.plot(time_steps, memory_max, label=model_name, color=colors[model_name], marker=markers[model_name], markersize=5, markevery=int(spacing[model_name]*T/10))
                 
        #### Json saving       
        results_json[model_name] = { 'label':label,
                                'regret':results[model_name]['regret'].tolist(),
                                'reward':results[model_name]['reward'].tolist(),
                                'regret_mean':regret_mean.tolist(),
                                'regret_std':regret_std.tolist(),
                                'reward_mean':reward_mean.tolist(),
                                'reward_std':reward_std.tolist(),
                                'time_cpu_mean':time_cpu_mean.tolist(),
                                'time_wall_mean':time_wall_mean.tolist(),
                                'memory_max':memory_max.tolist(),
                                'params':params_models[model_name]
                                }


    path_results = path_to_results(dataset, sub_PATH)


    if not os.path.exists(f'{PATH}/{path_results}'):
        os.makedirs(f'{PATH}/{path_results}')
                          
    ### Over T
    ax1.set_title(f'Cumulative Regret')
    ax1.set_xlabel('T')
    ax1.set_xlim(left=1)
    ax1.set_ylim(bottom=1)
    ax1.set_ylabel('Cumulative Regret')
    ax1.grid(linestyle = '--', linewidth = 0.5)
    ax1.legend()

    ax2.set_title(f'Cumulative Reward')
    ax2.set_xlabel('T')
    ax2.set_ylabel('Cumulative Reward')
    ax2.grid(linestyle = '--', linewidth = 0.5)
    ax2.legend()  

    ax3.set_title(f'CPU Time')
    ax3.set_yscale('log')
    ax3.set_xlabel('T')
    ax3.set_ylabel('Time (sec)')
    ax3.grid(linestyle = '--', linewidth = 0.5)
    ax3.legend()

    ax4.set_title(f'Peak Memory Usage')
    ax4.set_xlabel('T')
    ax4.set_ylabel('Memory (MiB)')
    ax4.grid(linestyle = '--', linewidth = 0.5)
    ax4.legend()

    ##### Check path 
    PATH_json = f'{PATH}/{dataset}/{sub_PATH_model}/json/'
    PATH_image = f'{PATH}/{dataset}/{sub_PATH_model}/images/'
    if not os.path.exists(PATH_json):
        os.makedirs(PATH_json)
    if not os.path.exists(PATH_image):
        os.makedirs(PATH_image)

    # Image saving
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(pad=2.0)
    fig.savefig(PATH_image + title + '.png', dpi=500)

    # Json saving
    dict_res = {'nb_runs':params_exp['nb_runs'],
                'loop_RP':params_exp['loop_RP'] if 'loop_RP' in params_exp else None,
                'seed_dataset':params_exp['seed_dataset'] if 'seed_dataset' in params_exp else None,
                'sigma':sigma, 
                'T':T, 
                'nb_actions':nb_actions, 
                'd':d, 
                'results':results_json}
    json.dump(dict_res, open(PATH_json + title + '.json', 'w'))

    return path_results


def plot_best_results(PATH, dataset, k=projected_dim, m=sketch_dim, box_plot=False):
    """
        Plot best results in terms of cumulative regret (smaller value), cpu and wall time for 
      each model given a dataset and sketch/projection dimesions
    """
    # Get models to plot/ not to plot
    list_models_to_not_plot = list_models_to_not_plot_s[dataset]
    
    # Possible values of k and m
    sketched_dim = list(np.arange(10, 205, 10))
    sketched_dim_copy = sketched_dim.copy()

    # Get list of models in specified path
    models_names = [f for f in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, f))]
    if 'ConfidenceBall1' in models_names and 'LinUCB' in models_names and 'Random' in models_names:
        idx = models_names.index('ConfidenceBall1')
        models_names[idx], models_names[-1] = models_names[-1], models_names[idx]
        idx = models_names.index('LinUCB')
        models_names[idx], models_names[-2] = models_names[-2], models_names[idx]
        idx = models_names.index('Random')
        models_names[idx], models_names[-3] = models_names[-3], models_names[idx]
    T=None
    nb_actions=None
    box_plot = []
    models_box_plot = []

    # Init variables
    regret_mean_over_t = {model_name: [] for model_name in models_names}
    regret_std_over_t = {model_name: [] for model_name in models_names}
    time_cpu_over_t = {model_name: [] for model_name in models_names}
    time_wall_over_t = {model_name: [] for model_name in models_names}
    reward_mean_over_t = {model_name: [] for model_name in models_names}
    reward_std_over_t = {model_name: [] for model_name in models_names}
    memory_over_t = {model_name: [] for model_name in models_names}
    label_over_t = {model_name: [] for model_name in models_names}
    memory_init = {model_name: 0 for model_name in models_names}

    # Scatter plot
    scatter = {model_name:{'x':[],
                           'y':[],
                           'yerr':[]}
               for model_name in models_names}

    scatter_dims = {'CBRAP':[200, 100], 'ConfidenceBall1_FJLT':[200, 100],
                    'CBSCFD':[50, 30], 'SOFUL_2m':[50,  30] }
    scatter_T_interval = np.array([ 0, -200,])
    x_param = 'time'



    for model_name in models_names:
        ##### Baseline models (that don't change with parameter m or k)
        if model_name in ['Random', 'LinUCB', 'ConfidenceBall1']:
            try: 
                PATH_ =  f'{PATH}{model_name}/json/'
                best_file = sorted(gb.glob(PATH_ + '*.json'),  key=numerical_sort)[0]
                f = open(best_file)
                data = json.load(f)
                T, d, sigma, nb_actions = data['T'], data['d'], data['sigma'], data['nb_actions']
                data = data['results']

                # Store values
                regret_mean_over_t[model_name] = np.array(data[model_name]['regret_mean'])
                regret_std_over_t[model_name] = np.array(data[model_name]['regret_std'])
                reward_mean_over_t[model_name] = np.array(data[model_name]['reward_mean'])
                reward_std_over_t[model_name] = np.array(data[model_name]['reward_std'])
                key_time_cpu = 'time_cpu_mean' if 'time_cpu_mean' in data[model_name].keys() else 'time_mean'
                time_cpu_over_t[model_name] = np.array(data[model_name][key_time_cpu])  
                time_wall_over_t[model_name] = np.array(data[model_name]['time_wall_mean']) if 'time_wall_mean' in data[model_name].keys() else None
                label_over_t[model_name] = data[model_name]['label']
                memory_over_t[model_name] = np.array(data[model_name]['memory_max'])
                memory_init[model_name] = memory_over_t[model_name][0]

                # For box plot
                if 'regret' in data[model_name].keys():
                    if model_name not in list_models_to_not_plot:   
                        box_plot.append(np.array(data[model_name]['regret'])[:,-1])
                        models_box_plot.append(model_name)
                
                # For scatter
                T_s = T + scatter_T_interval
                for t in T_s:
                    x = time_cpu_over_t[model_name]  if x_param=='time' else memory_over_t[model_name] 
                    scatter[model_name]['x'].append(x[t-1])
                    scatter[model_name]['y'].append(regret_mean_over_t[model_name][t-1])
                    scatter[model_name]['yerr'].append(regret_std_over_t[model_name][t-1])


                f.close()
            except Exception as e:
                print(model_name, e)
                pass

        ###### Other models
        else:
            # Figure 3: plot for each algo with multiple values of k and m
            fig_dim, (ax1_dim, ax2_dim, ax3_dim) = plt.subplots(1, 3, figsize=(16, 5))

            box_plot_dim, models_box_plot_dim = [], []
            for new_dim in sketched_dim_copy:
                try:
                    PATH_ =  f'{PATH}{model_name}/{new_dim}/json/'
                    best_file = sorted(gb.glob(PATH_ + '*.json'),  key=numerical_sort)[0]
                    f = open(best_file)
                    data = json.load(f)
                    T, d, sigma, nb_actions = data['T'], data['d'], data['sigma'], data['nb_actions']
                    data = data['results']

                    ### For plotting for various m or k
                    model_label = data[model_name]['label']
                    regret_mean_model = np.array(data[model_name]['regret_mean'])
                    regret_std_model = np.array(data[model_name]['regret_std'])
                    key_time_cpu = 'time_cpu_mean' if 'time_cpu_mean' in data[model_name].keys() else 'time_mean'
                    time_cpu_model = np.array(data[model_name][key_time_cpu])
                    time_wall_model = np.array(data[model_name]['time_wall_mean']) if 'time_wall_mean' in data[model_name].keys() else None
                    reward_mean_model = np.array(data[model_name]['reward_mean'])
                    reward_std_model = np.array(data[model_name]['reward_std'])
                    memory_model = np.array(data[model_name]['memory_max'])
                    memory_init[model_name] = memory_model[0]

                    # Figure 3
                    ## Regret mean plot
                    ax1_dim.plot(np.arange(T), regret_mean_model, label=model_label,  marker=markers[model_name], markersize=5, markevery=int(spacing[model_name]*T/10), alpha=0.9)
                    ## Save for boxplot
                    if 'regret' in data[model_name].keys():
                        box_plot_dim.append(np.array(data[model_name]['regret'])[:,-1])
                    ## Cpu time
                    ax3_dim.plot(np.arange(T), time_cpu_model, label=model_label, marker=markers[model_name], markersize=5, markevery=int(spacing[model_name]*T/10), alpha=0.9)
                    try:
                        models_box_plot_dim.append(f"m={data[model_name]['params']['m']}")
                    except:
                        models_box_plot_dim.append(f"k={data[model_name]['params']['k']}")

                    corresponding_dim = k if model_name in ['CBRAP', 'ConfidenceBall1_FJLT'] else m
                    if new_dim == corresponding_dim:

                        ### Plot over t with chosen m
                        regret_mean_over_t[model_name] = regret_mean_model
                        regret_std_over_t[model_name] = regret_std_model
                        reward_mean_over_t[model_name] = reward_mean_model
                        reward_std_over_t[model_name] =  reward_std_model
                        time_cpu_over_t[model_name] = time_cpu_model
                        time_wall_over_t[model_name] = time_wall_model
                        label_over_t[model_name] = model_label
                        memory_over_t[model_name] = memory_model

                        # For box plot
                        if model_name not in list_models_to_not_plot and 'regret' in data[model_name].keys():
                            box_plot.append(np.array(data[model_name]['regret'])[:,-1])
                            models_box_plot.append(model_name)
                    # For scatter
                    if new_dim in scatter_dims[model_name]:
                        T_s = T + scatter_T_interval
                        for t in T_s:
                            x = time_cpu_model  if x_param=='time' else memory_model
                            scatter[model_name]['x'].append(x[t-1])
                            scatter[model_name]['y'].append(regret_mean_model[t-1])
                            scatter[model_name]['yerr'].append(regret_std_model[t-1])


                    f.close()
                except Exception as e:
                    print(model_name, new_dim, e)
                    if new_dim in sketched_dim:
                        sketched_dim.remove(new_dim)

            # Figure 2
            if box_plot_dim != [] and len(box_plot_dim) == len(models_box_plot_dim):
                ## regret
                ax1_dim.set_title(f'Cumulative Regret')
                ax1_dim.set_xlabel('T')
                ax1_dim.set_ylabel('Cumulative Regret')
                ax1_dim.set_xlim(left=1)
                ax1_dim.set_ylim(bottom=1)
                ax1_dim.grid(linestyle = '--', linewidth = 0.5)
                ax1_dim.legend()  
    
                ## Boxplot
                ax2_dim.boxplot(box_plot_dim, showfliers=True)
                # print(models_box_plot_dim)
                ax2_dim.set_xticklabels(models_box_plot_dim)
                ax2_dim.set_xlabel('Models')
                ax2_dim.set_ylabel('Cumulative regret')
                ax2_dim.set_title('Cumulative regret at the end of iterations')
                ax2_dim.grid(linestyle = '--', linewidth = 0.5)

                ## cpu time
                ax3_dim.set_title(f'CPU Time')
                ax3_dim.set_xlabel('T')
                ax3_dim.set_ylabel('Time (sec)')
                # ax3_dim.set_yscale('log')
                ax3_dim.grid(linestyle = '--', linewidth = 0.5)
                ax3_dim.legend()

                fig_dim.tight_layout(pad=3)
                fig_dim.savefig(f'{PATH}{model_name}/{model_name}plot_for_various_reduced_dimension.pdf', format="pdf")



    ######################## Plots ########################
    # Figure 1: Plot over iterations
    fig_all, (ax1_all, ax2_all, ax3_all) = plt.subplots(1, 3, figsize=(16, 5))

    # Get max cpu time
    times = []
    for model in ['CBRAP', 'ConfidenceBall1_FJLT', 'CBSCFD', 'SOFUL_2m']:
        if model in time_cpu_over_t.keys():
            times.append(time_cpu_over_t[model][~np.isnan(time_cpu_over_t[model])][-1])
    max_time = max(times) if times != [] else float('+inf')
    time_step = np.arange(T)
    for _, model_name in enumerate(models_names):
        try:
            ##################### Figure1: Figure plot over T
            # Cut plots if algorithm didn't stop at time limit
            if dataset != 'MNIST':
                time_wall_over_t[model_name][time_cpu_over_t[model_name] > max_time] = np.nan
                regret_mean_over_t[model_name][time_cpu_over_t[model_name] > max_time] = np.nan
                time_cpu_over_t[model_name][time_cpu_over_t[model_name] > max_time] = np.nan

            if model_name not in list_models_to_not_plot:
                ax1_all.plot(time_step, regret_mean_over_t[model_name], label=label_over_t[model_name], color=colors[model_name], marker=markers[model_name], markersize=5, markevery=int(spacing[model_name]*T/10), alpha=0.9)
                # ax1_all.fill_between(time_step, np.array(regret_mean_over_t[model_name]) - np.array(regret_std_over_t[model_name]), 
                #                                  np.array(regret_mean_over_t[model_name]) + np.array(regret_std_over_t[model_name]), alpha=0.2, color=colors[model_name])

            ### CPU time
            idx_last = np.sum(~np.isnan(time_cpu_over_t[model_name]))-1
            time_last = np.round(time_cpu_over_t[model_name][~np.isnan(time_cpu_over_t[model_name])][-1], decimals=2)
            ax2_all.plot(time_step, time_cpu_over_t[model_name], label=f'{label_over_t[model_name]} | {time_last}s', color=colors[model_name], marker=markers[model_name], markersize=5, markevery=int(spacing[model_name]*T/10), alpha=0.9)
            ax2_all.scatter(idx_last, time_last, marker='X', color=colors[model_name])
            
            ### Wall time
            idx_last = np.sum(~np.isnan(time_wall_over_t[model_name]))-1
            time_last = np.round(time_wall_over_t[model_name][~np.isnan(time_wall_over_t[model_name])][-1], decimals=2)
            ax3_all.plot(time_step, time_wall_over_t[model_name], label=f'{label_over_t[model_name]} | {time_last}s', color=colors[model_name], marker=markers[model_name], markersize=5, markevery=int(spacing[model_name]*T/10), alpha=0.9)
            ax3_all.scatter(idx_last, time_last, marker='X', color=colors[model_name])


        except Exception as e:
            print(e)
            pass

    #### Figure 2: Box plot
    fig_box_plot, ax= plt.subplots(1, 1, figsize=(6, 5))

    ax.boxplot(box_plot, showfliers=False)
    ax.set_xlabel('Models')
    ax.set_ylabel('Cumulative regret')
    ax.set_title('Cumulative regret at the end of iterations')
    ax.set_xticklabels(models_box_plot)
    ax.grid(linestyle = '--', linewidth = 0.5)
    # fig_box_plot.savefig(PATH + 'Box_plot.pdf', format="pdf")

    title = f'{dataset}_d={d}_T={T}'
    if dataset =='Random':
        title += f'_actions={nb_actions}_σ={sigma}'


    #### Figure 1
    ax1_all.set_title(f'Cumulative Regret')
    ax1_all.set_xlabel('T')
    ax1_all.set_ylabel('Cumulative Regret')
    # ax1_all.set_yscale('log')
    ax1_all.set_xlim(left=1)
    ax1_all.set_ylim(bottom=1)
    ax1_all.grid(linestyle = '--', linewidth = 0.5)
    ax1_all.legend()  

    ax2_all.set_title(f'CPU Time')
    ax2_all.set_xlabel('T')
    ax2_all.set_ylabel('Time (sec)')
    ax2_all.set_yscale('log')
    ax2_all.grid(linestyle = '--', linewidth = 0.5)
    ax2_all.legend()  

    ax3_all.set_title(f'Wall Time')
    ax3_all.set_xlabel('T')
    ax3_all.set_ylabel('Time (sec)')
    ax3_all.set_yscale('log')
    ax3_all.grid(linestyle = '--', linewidth = 0.5)
    ax3_all.legend()  

    fig_all.tight_layout(pad=3)
    fig_all.suptitle(title, fontsize=14)
    fig_all.savefig(PATH + 'Plot_over_time.pdf', format="pdf")


    #### Figure 3: Scatter
    fig_scatter, ax = plt.subplots(1, 1, figsize=(7, 4))
    max_y_lim, min_y_lim = 0, 0
    # Create a scatter plot with error bars
    for model_name in models_names:
        try:
            ax.errorbar(**scatter[model_name], fmt='o', color=colors[model_name], ecolor=colors[model_name], alpha=0.6, capsize=5, label=model_name)
            max_y_lim = max(scatter[model_name]['yerr']) + 150 if  max(scatter[model_name]['yerr']) + 150 > max_y_lim else max_y_lim
            min_y_lim = min(scatter[model_name]['yerr']) - 150 if  min(scatter[model_name]['yerr']) - 150 < min_y_lim else min_y_lim
        except Exception as e:
            pass
    # Customize plot labels and title
    xlabel = 'CPU Time (sec)' if x_param == 'time' else 'Peak Memory (MiB)'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative regret')
    ax.set_xscale('log')
    # ax.axis(ymin=min_y_lim, ymax=max_y_lim)
    ax.grid(linestyle = '--', linewidth = 0.5)
    ax.set_title(f'{dataset} - Performance comparison for {x_param}')
    ax.legend()

    fig_scatter.savefig(PATH + 'scatter.pdf', format="pdf")




    # #### Scatter interactive plot
    # if plot_3d:
    #     for key in ['Random', 'LinUCB', 'ConfidenceBall1']:
    #         if key in scatter:
    #             del scatter[key]
    #     df = pd.DataFrame.from_dict(scatter, orient='index')
    #     df['model'] = df.index
    #     df = df.explode(['CPU_Time (s)', 'Memory_Peak (MiB)', 'Cumulative_Regret', 'yerr'])
    #     df = df.explode(['CPU_Time (s)', 'Memory_Peak (MiB)', 'Cumulative_Regret', 'yerr'])
    #     fig = px.scatter_3d(df, x='CPU_Time (s)', log_x=True,
    #                             y='Memory_Peak (MiB)', 
    #                             z='Cumulative_Regret', error_z='yerr',
    #                             color='model')
    #     fig.update_scenes(xaxis_autorange="reversed")
    #     fig.update_scenes(yaxis_autorange="reversed")
    #     fig.show()



def plot_over_param(PATH, dataset, param='scale', k=projected_dim, m=sketch_dim,):
    """
        Plot cumulative regret variations for each model over a certain parameter
    """
    ### Get list of models in specified path
    models_names = [f for f in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, f))]
    if 'ConfidenceBall1' and 'LinUCB' in models_names:
        idx = models_names.index('ConfidenceBall1')
        models_names[idx], models_names[-1] = models_names[-1], models_names[idx]
        idx = models_names.index('LinUCB')
        models_names[idx], models_names[-2] = models_names[-2], models_names[idx]
    T=None
    nb_actions=None



    ### For plots over param
    regret_mean_over_param = {model_name: [] for model_name in models_names}
    regret_std_over_param = {model_name: [] for model_name in models_names}
    time_cpu_over_param = {model_name: [] for model_name in models_names}
    reward_mean_over_param = {model_name: [] for model_name in models_names}
    reward_std_over_param = {model_name: [] for model_name in models_names}
    memory_over_param = {model_name: [] for model_name in models_names}
    label_over_param = {model_name: [] for model_name in models_names}


    sketched_dim = list(np.arange(5, 105, 5))
    lam_s = [2*(10**i) for i in range(-11, 8)]
    scale_s = [(10**i) for i in range(-6, 5)] 

    params_values = None
    models_fixed_names = None
    if param == 'm' or param == 'k':
        params_values = sketched_dim
        models_fixed_names = intersect(models_names, ['Random', 'LinUCB', 'ConfidenceBall1'])
    elif param == 'lam':
        models_fixed_names = intersect(models_names, ['Random', 'LinUCB', 'ConfidenceBall1'])
        params_values = lam_s
    elif param == 'scale':
        models_fixed_names = intersect(models_names, ['Random', 'LinUCB', 'ConfidenceBall1'])
        params_values = scale_s
    for model in models_fixed_names:            
        if model in models_names:  
            models_names.remove(model)  


    results = {model:{
                        p: {
                            'regret_mean': None,
                            'regret_std': None,
                            'reward_mean': None,
                            'reward_std': None,
                            'time_cpu': None,
                            'memory': None,
                        } for p in params_values
                    } for model in models_names}
    for model_name in models_names + models_fixed_names: 
        ##### Models that doesn't change with the parameter
        if model_name in models_fixed_names:
            try: 
                PATH_ =  f'{PATH}{model_name}/json/'
                best_file = sorted(gb.glob(PATH_ + '*.json'),  key=numerical_sort)[0]
                f = open(best_file)
                data = json.load(f)
                T, d, sigma, nb_actions = data['T'], data['d'], data['sigma'], data['nb_actions']
                data = data['results']


                # For plotting over param
                regret_mean_over_param[model_name] = [data[model_name]['regret_mean'][-1]]
                regret_std_over_param[model_name] = [data[model_name]['regret_std'][-1]]
                key_time_cpu = 'time_cpu_mean' if 'time_cpu_mean' in data[model_name].keys() else 'time_mean'
                time_cpu_over_param[model_name] = [data[model_name][key_time_cpu][-1]]
                reward_mean_over_param[model_name] = [data[model_name]['reward_mean'][-1]]
                reward_std_over_param[model_name] = [data[model_name]['reward_std'][-1]]
                memory_over_param[model_name] = [data[model_name]['memory_max'][-1]]

                f.close()
            except Exception as e:
                print(model_name, e)
                pass
        ###### Other models
        else:
            if param == 'm' or param == 'k':
                for p in params_values:
                    try:
                        PATH_ =  f'{PATH}{model_name}/{p}/json/'
                        best_file = sorted(gb.glob(PATH_ + '*.json'),  key=numerical_sort)[0]
                        f = open(best_file)
                        data = json.load(f)
                        T, d, sigma, nb_actions = data['T'], data['d'], data['sigma'], data['nb_actions']
                        data = data['results']

                        ### For plotting over m
                        results[model_name][p]['regret_mean'] = np.array(data[model_name]['regret_mean'])[-1]
                        results[model_name][p]['regret_std'] = np.array(data[model_name]['regret_std'])[-1]
                        results[model_name][p]['reward_mean'] = np.array(data[model_name]['reward_mean'])[-1]
                        results[model_name][p]['reward_std'] = np.array(data[model_name]['reward_std'])[-1]
                        key_time_cpu = 'time_cpu_mean' if 'time_cpu_mean' in data[model_name].keys() else 'time_mean'  
                        results[model_name][p]['time_cpu'] = np.array(data[model_name][key_time_cpu])[-1]
                        results[model_name][p]['memory'] = max(np.array(data[model_name]['memory_max'])[1:])
                        label_over_param[model_name] = data[model_name]['label']

                        f.close()
                    except Exception as e:
                        pass
            else:
                try:
                    corresponding_dim = k if model_name in ['CBRAP', 'ConfidenceBall1_FJLT'] else m
                    PATH_ =  f'{PATH}{model_name}/{corresponding_dim}/json/'
                    files = sorted(gb.glob(PATH_ + '*.json'),  key=numerical_sort)
                    best_file = files[0]
                    f = open(best_file)
                    data = json.load(f)
                    T, d, sigma, nb_actions = data['T'], data['d'], data['sigma'], data['nb_actions']
                    best_params = data['results'][model_name]['params']
                    f.close()
                    for file in files:
                        f = open(file)
                        data_other = json.load(f)
                        data = data_other['results'][model_name]
                        load_file = []
                        for param_name in data_other['results'][model_name]['params'].keys():
                            if param_name != param:
                                load_file.append(best_params[param_name] == data_other['results'][model_name]['params'][param_name])                        
                        load_file = all(load_file)
                        if load_file:
                            
                            results[model_name][float(data['params'][param])]['regret_mean'] = np.array(data['regret_mean'])[-1]
                            results[model_name][float(data['params'][param])]['regret_std'] = np.array(data['regret_std'])[-1]
                            results[model_name][float(data['params'][param])]['reward_mean'] = np.array(data['reward_mean'])[-1]
                            results[model_name][float(data['params'][param])]['reward_std'] = np.array(data['reward_std'])[-1]
                            results[model_name][float(data['params'][param])]['time_cpu_mean'] = np.array(data['time_cpu_mean'])[-1]
                            results[model_name][float(data['params'][param])]['memory'] = max(np.array(data['memory_max'])[1:])
                            label_over_param[model_name] = data['label']

                        f.close()
                except Exception as e:
                    pass
    # Get the values of the parameter in common for all models
    all_values = []
    for model_name in models_names:
        vals = []
        for p in params_values:
            if results[model_name][p]['regret_mean'] is not None:
                vals.append(p)
        all_values.append(vals)
    print(models_names)
    print(all_values)
    vals_in_common = reduce(np.intersect1d, all_values)

    # Keep only results of params in common 
    for val in vals_in_common:
        for model_name in models_names:
            regret_mean_over_param[model_name].append(results[model_name][val]['regret_mean'])
            regret_std_over_param[model_name].append(results[model_name][val]['regret_std'])
            reward_mean_over_param[model_name].append(results[model_name][val]['reward_mean'])
            reward_std_over_param[model_name].append(results[model_name][val]['reward_std'])
            time_cpu_over_param[model_name].append(results[model_name][val]['time_cpu_mean'])
            memory_over_param[model_name].append(results[model_name][val]['memory'])




    ######################## Plots ########################
    fig, ax1 = plt.subplots(figsize=(6, 5))
    print('ici')
    print(vals_in_common)
    for model_name in models_names:
        try:
            #### Best over m
            # Regret
            ax1.plot(vals_in_common, regret_mean_over_param[model_name], label=label_over_param[model_name], marker=markers[model_name], markersize=5, color=colors[model_name])

            ax1.fill_between(vals_in_common, np.array(regret_mean_over_param[model_name]) - np.array(regret_std_over_param[model_name]), 
                                  np.array(regret_mean_over_param[model_name]) + np.array(regret_std_over_param[model_name]), alpha=0.2, color=colors[model_name])
            



        except Exception as e:
            print(e)
            pass

    title = f'{dataset}_n={T}'
    if dataset =='Random':
        title += f'_d={d}_actions={nb_actions}_σ={sigma}'

    ax1.set_title(f'{dataset} - Cumulative Regret Variations Over Parameter')
    ax1.set_xlabel(param)
    ax1.set_ylabel('Cumulative Regret')
    ax1.grid(linestyle = '--', linewidth = 0.5)
    ax1.set_xscale('log')
    # ax1.set_xlim(left=1)
    # ax1.set_ylim(bottom=1)
    ax1.legend() 



    fig.savefig(PATH + f'Plot_over_{param}.pdf', format="pdf")


def results_memory(PATH, k=50, m=10):
    """
        Get memory usage of each model given a dataset (Path)
    """
    ### Get list of models in specified path
    models_names = [f for f in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, f))]
    if 'ConfidenceBall1' in models_names and 'LinUCB' in models_names and 'Random' in models_names:
        idx = models_names.index('ConfidenceBall1')
        models_names[idx], models_names[-1] = models_names[-1], models_names[idx]
        idx = models_names.index('LinUCB')
        models_names[idx], models_names[-2] = models_names[-2], models_names[idx]
        idx = models_names.index('Random')
        models_names[idx], models_names[-3] = models_names[-3], models_names[idx]


    ### For plots over t
    label_over_t = {model_name: [] for model_name in models_names}
    max_memory = {model_name: 0 for model_name in models_names}
    init_memory = {model_name: 0 for model_name in models_names}

    sketched_dim = list(np.arange(5, 205, 5))
    sketched_dim_copy = sketched_dim.copy()

    for model_name in models_names:
        ##### Models that doesn't change with parameter m or k
        if model_name in ['Random', 'LinUCB', 'ConfidenceBall1']:
            try: 
                PATH_ =  f'{PATH}{model_name}/json/'
                best_file = sorted(gb.glob(PATH_ + '*.json'),  key=numerical_sort)[0]
                f = open(best_file)
                data = json.load(f)
                data = data['results']


                # For plotting over t
                label_over_t[model_name] = data[model_name]['label']
                memory = np.array(data[model_name]['memory_max'][1:])
                memory = memory[~np.isnan(memory)]
                max_memory[model_name] = np.max(memory)
                init_memory[model_name] = data[model_name]['memory_max'][0]


                f.close()
            except Exception as e:
                print(model_name, e)
                pass
        ###### Other models
        else:
            for new_dim in sketched_dim_copy:
                try:
                    PATH_ =  f'{PATH}{model_name}/{new_dim}/json/'
                    best_file = sorted(gb.glob(PATH_ + '*.json'),  key=numerical_sort)[0]
                    f = open(best_file)
                    data = json.load(f)
                    data = data['results']

                    ### For plotting over m
                    model_label = data[model_name]['label']
                    memory_model = np.max(data[model_name]['memory_max'][1:])
                    init_memory[model_name] = data[model_name]['memory_max'][0]

                    corresponding_dim = k if model_name in ['CBRAP', 'ConfidenceBall1_FJLT'] else m
                    if new_dim == corresponding_dim:
                        label_over_t[model_name] = model_label
                        max_memory[model_name] = memory_model
                        print(model_name, memory_model)

                    f.close()
                except Exception as e:
                    if new_dim in sketched_dim:
                        sketched_dim.remove(new_dim)

    dict_mem = {'init_mem': init_memory, 'max_mem': max_memory}
    return dict_mem


def plot_memory(results, PATH='./', init=False):
    """
        Bar plot of memory usage for each dataset and model in Path
    """
    label = 'init_mem' if init else 'max_mem'
    title_file = 'Initial' if init else 'Maximum'
    title = 'at initialization' if init else 'during iterations'

    models_dataset = [list(results[dataset][label].keys()) for dataset in results.keys()]
    datasets = results.keys()
    models = reduce(np.intersect1d, models_dataset)
    size = 0.08
    spc = - size * (len(models)/2)
    print(spc)
    datasets_idx = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(10, 5))

    res = {}
    sorted_res = {}
    for i, dataset in enumerate(datasets):
        res[dataset] = {}
        for model in models:
            res[dataset][model] = results[dataset][label][model]
        sorted_res[dataset] = dict(sorted(res[dataset].items(), key=lambda item: item[1], reverse=True))
    spc_idx = np.array([(spc + i*size)-(size/2) for i in range(1, len(models)+1)])
 

    for _, model in enumerate(models):
        x = [ i + spc_idx[list(sorted_res[dataset].keys()).index(model)]  for i, dataset in enumerate(datasets)]
        values = [sorted_res[dataset][model] for dataset in sorted_res.keys()]
        print(model, values)
        ax.bar(x, values, size, label = model, alpha=0.75, color=colors[model], hatch=hatchs[model])


    
    ax.set(xticks=datasets_idx, xticklabels=datasets)
    ax.set_ylabel("Peak memory (MiB)") 
    ax.set_yscale("log")
    ax.set_title("Peak of memory allocation " + title) 
    ax.grid(linestyle = '--', linewidth = 0.5)
    # ax.legend() 
    ax.legend(loc='upper center',  bbox_to_anchor=(0.5, -0.05), ncol=len(models), fancybox=True, shadow=True)
    fig.savefig(PATH + title_file + '_memory_test.pdf', format='pdf')


def latex_memory(results, PATH='./', init=False):
    """
        Latex table describing memory usage for each dataset and model in Path
    """
    label = 'init_mem' if init else 'max_mem'
    title_file = 'Initial' if init else 'Maximum'
    title = 'at initialization' if init else 'during iterations'

    models = list(results[list(results.keys())[0]][label].keys())

    string = "\\begin{table}[H]\n \\centering\n \\begin{tabular}{"
    for _ in range(len(list(results.keys())) + 1):
        string += '|l'
    string += '|} \\hline\n  \\backslashbox{\\textbf{Algorithm}}{\\textbf{Dataset}}'
    for dataset in list(results.keys()):
        string += '& \\textbf{' + dataset +'}'
    string += '\\\ \\hline\n  '
    for model in models:
        string += model.replace('_2m', '').replace('_', '\_')
        for dataset in list(results.keys()):
            string += rf'  & {results[dataset][label][model]}'
        string += '\\\ \\hline\n  '
    string += '\n \\end{tabular}\n \\caption{Peak of memory allocation ' + title +'}\n\\end{table}'
    f = open(PATH + title_file + "_memory.txt", "w")
    f.write(string)
    f.close()
        