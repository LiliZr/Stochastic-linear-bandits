import glob
import json
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px


dataset = 'MNIST'
m_actions = 10000
d = 4097
n = 4000
sigma = 0.3
sub_path = f'd={d}_σ={sigma}_actions={m_actions}_n={n}/' if dataset =='Random' else ''

PATH = f'/home/lizri/phd/code1/stochastic-linear-bandits/results/parameters/{dataset}/{sub_path}'

m_s = [10, 50]


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts



models_names = ['Random',  'CBSCFD', 'LinREL1_FJLT', 'LinUCB', 'LinREL1']
regret_mean = {model_name: [] for model_name in models_names}
regret_std = {model_name: [] for model_name in models_names}
running_time = {model_name: [] for model_name in models_names}
reward_mean = {model_name: [] for model_name in models_names}
reward_std = {model_name: [] for model_name in models_names}
memory = {model_name: [] for model_name in models_names}

##### For scatter plot
# scatter = {name :{'Memory_Peak (MiB)':[],
#                 'CPU_Time (s)': [],
#                 'Cumulative_Regret':[],
#                 } for name in models_names}

scatter = {name :{'Memory_Peak (MiB)':[],
                'CPU_Time (s)': [],
                'Cumulative_Regret':[],
                'yerr':[]
                } for name in models_names}

##### For "m" comparison 
m_model = {'LinREL1_FJLT': 50,
           'CBSCFD': 10}
regret_mean_m = {model_name: [] for model_name in models_names}
regret_std_m = {model_name: [] for model_name in models_names}
running_time_m = {model_name: [] for model_name in models_names}
reward_mean_m = {model_name: [] for model_name in models_names}
reward_std_m = {model_name: [] for model_name in models_names}
label_m = {model_name: [] for model_name in models_names}
memory_m = {model_name: [] for model_name in models_names}
memory_init = {model_name: 0 for model_name in models_names}

m_s_copy = m_s.copy()

###### CBSCFD to remove
running_time_100 = None
label_m_100 = None

for model_name in models_names:
    ##### Models that doesn't change with parameter
    if model_name in ['Random', 'LinUCB', 'LinREL1']:
        try: 
            PATH_ =  f'{PATH}{model_name}/json/'
            best_file = sorted(glob.glob(PATH_ + '*.json'),  key=numericalSort)[0]
            f = open(best_file)
            data = json.load(f)
            data = data['results']


            regret_mean[model_name] = [data[model_name]['regret_mean'][-1]] * len(m_s)
            regret_std[model_name] = [data[model_name]['regret_std'][-1]] * len(m_s)
            running_time[model_name] = [data[model_name]['time_mean'][-1]] * len(m_s)
            reward_mean[model_name] = [data[model_name]['reward_mean'][-1]] * len(m_s)
            reward_std[model_name] = [data[model_name]['reward_std'][-1]] * len(m_s)
            memory[model_name] = [data[model_name]['memory_max'][-1]] * len(m_s)


            # scatter[model_name]['y'].append(data[model_name]['regret_mean'][-1])
            # scatter[model_name]['yerr'].append(data[model_name]['regret_std'][-1])
            # scatter[model_name]['x'].append(data[model_name]['time_mean'][-1])

            scatter[model_name]['Cumulative_Regret'].append(data[model_name]['regret_mean'][-1])
            scatter[model_name]['yerr'].append(data[model_name]['regret_std'][-1])
            scatter[model_name]['CPU_Time (s)'].append(data[model_name]['time_mean'][-1])
            scatter[model_name]['Memory_Peak (MiB)'].append(max(data[model_name]['memory_max'][1:]))


            regret_mean_m[model_name] = np.array(data[model_name]['regret_mean'])
            regret_std_m[model_name] = np.array(data[model_name]['regret_std'])
            reward_mean_m[model_name] = np.array(data[model_name]['reward_mean'])
            reward_std_m[model_name] = np.array(data[model_name]['reward_std'])
            running_time_m[model_name] = np.array(data[model_name]['time_mean'])
            label_m[model_name] = data[model_name]['label']
            memory_m[model_name] = np.array(data[model_name]['memory_max'])
            memory_init[model_name] = memory_m[model_name][0]
            f.close()
        except:
            print(model_name)
            pass
    ###### Other models
    else:
        for m in m_s_copy:
                # print(model_name, m)
            try:
                PATH_ =  f'{PATH}{model_name}/{m}/json/'
                best_file = sorted(glob.glob(PATH_ + '*.json'),  key=numericalSort)[0]

                f = open(best_file)
                # current model
                data = json.load(f)
                data = data['results']

                ### plot Over m

                model_label = data[model_name]['label']
                regret_mean_model = np.array(data[model_name]['regret_mean'])
                regret_std_model = np.array(data[model_name]['regret_std'])
                running_time_model = np.array(data[model_name]['time_mean'])
                reward_mean_model = np.array(data[model_name]['reward_mean'])
                reward_std_model = np.array(data[model_name]['reward_std'])
                memory_model = np.array(data[model_name]['memory_max'])
                memory_init[model_name] = memory_model[0]
                ### Plot over t with chosen m
                if m == m_model[model_name]:

                    regret_mean[model_name].append(regret_mean_model[-1])
                    regret_std[model_name].append(regret_std_model[-1])
                    reward_mean[model_name].append(reward_mean_model[-1])
                    reward_std[model_name].append(reward_std_model[-1])
                    running_time[model_name].append(running_time_model[-1])
                    memory[model_name].append(max(memory_model[1:]))


                    ### Scatter

                    # scatter[model_name]['y'].append(regret_mean_model[-1])
                    # scatter[model_name]['yerr'].append(regret_std_model[-1])
                    # scatter[model_name]['x'].append(running_time_model[-1])

                    scatter[model_name]['Cumulative_Regret'].append(regret_mean_model[-1])
                    scatter[model_name]['yerr'].append(regret_std_model[-1])
                    scatter[model_name]['CPU_Time (s)'].append(running_time_model[-1])
                    scatter[model_name]['Memory_Peak (MiB)'].append(max(memory_model[1:]))


                # ### Plot over t with chosen m
                # if m == m_model[model_name]:
                    regret_mean_m[model_name] = regret_mean_model
                    regret_std_m[model_name] = regret_std_model
                    reward_mean_m[model_name] = reward_mean_model
                    reward_std_m[model_name] =  reward_std_model
                    running_time_m[model_name] = running_time_model
                    label_m[model_name] = model_label
                    memory_m[model_name] = memory_model
                if model_name=='CBSCFD' and m==100:
                    running_time_100 = running_time_model[-1]
                    label_m_100 = model_label

                f.close()
            except Exception as e:
                print(model_name, m, e)
                if m in m_s:
                    m_s.remove(m)


######################## Plots ########################
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))

fig_memory, (ax1_mem, ax2_mem) = plt.subplots(1, 2, figsize=(12, 6))
fig_all, ((ax1_all, ax2_all), (ax3_all, ax4_all)) = plt.subplots(2, 2, figsize=(13, 9))

time_step = np.arange(n)

print(memory_m)
for model_name in models_names:
    try:
        #### Best over m
        # Regret
        # ax1.plot(m_s, regret_mean[model_name], label=model_name, marker='o', markersize=4,)

        # ax1.fill_between(m_s, np.array(regret_mean[model_name]) - np.array(regret_std[model_name]), 
        #                       np.array(regret_mean[model_name]) + np.array(regret_std[model_name]), alpha=0.2)
        
        # # Reward
        # ax2.plot(m_s, reward_mean[model_name], label=model_name, marker='o', markersize=4,)

        # ax2.fill_between(m_s, np.array(reward_mean[model_name]) - np.array(reward_std[model_name]), 
        #                       np.array(reward_mean[model_name]) + np.array(reward_std[model_name]), alpha=0.2)
        # # Time
        # ax3.plot(m_s, running_time[model_name], label=model_name, marker='o', markersize=4,)
        # ax3.scatter(m_s[-1], running_time[model_name][-1], 
        #             label=np.round(running_time[model_name][-1], decimals=3), marker='x')
        # # Memory
        # ax4.plot(m_s, memory[model_name], label=model_name, marker='o', markersize=4,)
        # ax4.scatter(m_s[-1], memory[model_name][-1], 
        #             label=np.round(memory[model_name][-1], decimals=3), marker='x')
        

        ##################### Figure plot over T
        idx_last = np.sum(~np.isnan(memory_m[model_name]))-1
        print(model_name, memory_m[model_name])
        spc = int(time_step[-1]/10)
        memory_last = np.round(max(memory_m[model_name][~np.isnan(memory_m[model_name])][1:]), decimals=2)
        #### Combine best results over T
        if model_name!='LinUCB' and model_name!='LinREL1':
            ax1_all.plot(time_step, regret_mean_m[model_name], label=label_m[model_name], marker='o', markersize=4, markevery=int(time_step[-1]/10),)
            ax1_all.fill_between(time_step, np.array(regret_mean_m[model_name]) - np.array(regret_std_m[model_name]), 
                                            np.array(regret_mean_m[model_name]) + np.array(regret_std_m[model_name]), alpha=0.2)
            ax2_all.plot(time_step, reward_mean_m[model_name], label=label_m[model_name], marker='o', markersize=4, markevery=int(time_step[-1]/10),)
            ax2_all.fill_between(time_step, np.array(reward_mean_m[model_name]) - np.array(reward_std_m[model_name]), 
                                            np.array(reward_mean_m[model_name]) + np.array(reward_std_m[model_name]), alpha=0.2)
            print(model_name, type(memory_m[model_name]))
            spc = 10 if model_name == 'CBSCFD' else int(time_step[-1]/10)
            ax4_all.plot(time_step[1:], memory_m[model_name][1:], label=f'{label_m[model_name]} | {memory_last}', marker='o', markersize=4, markevery=spc, alpha=0.3)
            ax4_all.scatter(idx_last, memory_last, marker='x')

        idx_last = np.sum(~np.isnan(running_time_m[model_name]))-1
        time_last = np.round(running_time_m[model_name][~np.isnan(running_time_m[model_name])][-1], decimals=2)
        ax3_all.plot(time_step, running_time_m[model_name], label=f'{label_m[model_name]} | {time_last}', marker='o', markersize=4, markevery=int(time_step[-1]/10))
        ax3_all.scatter(idx_last, time_last, marker='x')
        ############################## Figure memory
        ax1_mem.plot(time_step[1:], memory_m[model_name][1:], label=f'{label_m[model_name]} | {memory_last}', marker='o', markersize=4, markevery=spc, alpha=0.3)
        ax2_mem.scatter(model_name, memory_init[model_name], label=f'{label_m[model_name]} | {memory_init[model_name]}', s=50)
        ax2_mem.stem(model_name, memory_init[model_name], markerfmt=' ', linefmt='C0-.')




    except Exception as e:
        print(e)
        pass
### TO remove
# ax3_all.scatter(time_step[-1], running_time_100, 
#             label=f'{label_m_100}_{np.round(running_time_100, decimals=3)}', marker='x')

title = f'{dataset}_n={n}'
if dataset =='Random':
    title += f'_d={d}_σ={sigma}'

############ Over m
ax1.set_title(f'- Cumulative Regret')
ax1.set_xlabel('m')
ax1.set_ylabel('Cumulative Regret')
ax1.grid(linestyle = '--', linewidth = 0.5)
ax1.legend() 

ax2.set_title(f'- Cumulative Reward')
ax2.set_xlabel('m')
ax2.set_ylabel('Cumulative Reward')
ax2.grid(linestyle = '--', linewidth = 0.5)
ax2.legend()  

ax3.set_title(f'CPU Time')
ax3.set_yscale('log')
ax3.set_xlabel('m')
ax3.set_ylabel('Time (sec)')
ax3.grid(linestyle = '--', linewidth = 0.5)
ax3.legend()

ax4.set_title(f'Peak Memory Usage')
ax4.set_xlabel('m')
ax4.set_ylabel('Memory (MiB)')
ax4.grid(linestyle = '--', linewidth = 0.5)
ax4.legend()  

# fig.savefig(PATH + 'Plot_over_m.png', dpi=500)

############# Over time
ax1_all.set_title(f'Cumulative Regret')
ax1_all.set_xlabel('n')
ax1_all.set_ylabel('Cumulative Regret')
ax1_all.grid(linestyle = '--', linewidth = 0.5)
ax1_all.legend()  

ax2_all.set_title(f'Cumulative Reward')
ax2_all.set_xlabel('n')
ax2_all.set_ylabel('Cumulative Reward')
ax2_all.grid(linestyle = '--', linewidth = 0.5)
ax2_all.legend()  

ax3_all.set_title(f'CPU Time')
ax3_all.set_yscale('log')
ax3_all.set_xlabel('n')
ax3_all.set_ylabel('Time (sec)')
ax3_all.grid(linestyle = '--', linewidth = 0.5)
ax3_all.legend()

ax4_all.set_title(f'Peak Memory Usage')
ax4_all.set_xlabel('n')
ax4_all.set_ylabel('Memory (MiB)')
ax4_all.grid(linestyle = '--', linewidth = 0.5)
ax4_all.legend()  


fig_all.suptitle(title, fontsize=14)
fig_all.savefig(PATH + 'Plot_over_time.png', dpi=500)
############# Mem
ax1_mem.set_title(f'Peak Memory Usage')
ax1_mem.set_xlabel('n')
ax1_mem.set_ylabel('Memory (MiB)')
ax1_mem.grid(linestyle = '--', linewidth = 0.5)
ax1_mem.legend()

ax2_mem.set_title(f'Peak Memory Usage at Initialization')
ax2_mem.set_xlabel('Model')
ax2_mem.set_ylabel('Memory (MiB)')
ax2_mem.grid(linestyle = '--', linewidth = 0.5)
ax2_mem.legend() 
fig_memory.savefig(PATH + 'Memory.png', dpi=500)

############# Scatter

# fig2 = plt.figure(figsize=(12, 6))
# ax = fig2.subplots()
# for model_name in models_names:
#     if scatter[model_name]['x'] != []:
#         ax.errorbar(**scatter[model_name],   label=model_name, fmt='o', capsize=3, alpha=0.5)
#     if model_name not in ['LinUCB', 'Random']:
#         for i, m in enumerate(m_s):
#             ax.annotate(m, (scatter[model_name]['x'][i], scatter[model_name]['y'][i]), size=6)
# title = f'Scatter Performance Comparison - d={d}'
# ax.legend()
# ax.set_title(title)
# ax.set_ylabel('Regret')
# ax.set_xlabel('CPU Time (sec)')
# ax.set_CPU_Time (s)cale('log')
# ax.grid(linestyle = '--', linewidth = 0.5)
# fig2.savefig(PATH + title + '.png', dpi=500)



########## Scatter 3d

# fig2 = plt.figure(figsize=(12, 6))
# ax = fig2.add_subplot(projection='3d')
# for model_name in ['LinREL1_FJLT', 'CBSCFD']:
#         print(scatter[model_name]['CPU_Time (s)'], scatter[model_name]['Memory_Peak (MiB)'], scatter[model_name]['Cumulative_Regret'],)
#         ax.scatter(**scatter[model_name],  label=model_name)
# title = f'Scatter Performance Comparison - d={d}'
# ax.legend()
# ax.set_title(title)
# ax.set_zlabel('Regret')
# ax.set_xlabel('CPU Time (sec)')
# ax.set_ylabel('Memory (MiB)')

# # ax.set_CPU_Time (s)cale('log')
# # ax.grid(linestyle = '--', linewidth = 0.5)
# fig2.savefig(PATH + title + '.png', dpi=500)


### v2
for key in ['LinUCB', 'LinREL1']:
    del scatter[key]
df = pd.DataFrame.from_dict(scatter, orient='index')
for column in df.columns:
    df[column] = df[column].apply(lambda x: x[0])
df['model'] = df.index
print(df)
fig = px.scatter_3d(df, x='CPU_Time (s)', y='Memory_Peak (MiB)', z='Cumulative_Regret', error_z='yerr',
              color='model')
fig.update_scenes(xaxis_autorange="reversed")
fig.update_scenes(yaxis_autorange="reversed")
fig.show()




