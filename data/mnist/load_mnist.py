import pandas as pd
import numpy as np

def load_mnist(path):
    df = pd.read_csv(f'{path}')
    label = df['5']
    df.drop('5', inplace=True, axis=1)
    action_set = {}
    nb_actions = 0
    d = 0 
    for i in range(10):
        mask = label == i
        action_set[str(i)] = np.array(df[mask])
        d = action_set[str(i)].shape[1]
        nb_actions += action_set[str(i)].shape[0]
    return action_set, nb_actions, d