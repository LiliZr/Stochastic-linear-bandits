from data.movielens_25m.load_movielens import *
from data.random.generateRandomGaussian import *
from data.mnist.load_mnist import * 
from data.steam.load_steam import *
from data.amazon.load_amazon import *
from data.yahoo.load_yahoo import *


def load_dataset(df_data=None, d=30, nb_actions=2000, name='Random', path='./data/', finite_set=True, seed=12, init=False):
    rng = np.random.RandomState(seed)
    theta, action_set = None, None
    if name == 'Random':
        theta = generate_theta(d, rng)
        if finite_set:
            action_set = generate_action_set(nb_actions, d, rng)
    elif name == 'MNIST':
        path += 'mnist/data/mnist.zip'
        action_set, nb_actions, d = load_mnist(path)
    elif name == 'Movielens':
        path += 'movielens_25m/data/'
        contextual_arms = np.load(path + 'contextual_arms.zip', allow_pickle=True)['contextual_arms'].astype(np.float64)
        rewards = np.load(path + 'rewards.zip', allow_pickle=True)['contextual_arms'].astype(np.float64)
        action_set = list(zip(contextual_arms, rewards))
        nb_actions, d = contextual_arms[0].shape 
    elif name == 'Steam':
        path += 'steam/data'
        recommendations = pd.read_csv(f'{path}/steam_links.zip')
        games_vectors = pd.read_csv(f'{path}/steam_actions.zip')
        action_set, nb_actions, d = load_steam(recommendations, games_vectors, seed=seed)
        for i, user in enumerate(action_set):
            print(i, np.sum(user[1]))
        print('####################')
    elif name == 'Yahoo':
        path += 'yahoo/data/yahoo.zip'
        reviews_x_items = pd.read_csv(path, sep=",", index_col=0)
        action_set, nb_actions, d = load_yahoo(reviews_x_items)

        nb_actions, d = action_set[0][0].shape 
    elif name == 'Amazon':
        path += 'amazon/data/amazon.zip'
        if init:
           reviews_x_items = pd.read_csv(path, sep=",", index_col=0)
           return [reviews_x_items]
        else:
            reviews_x_items = df_data[0]
            theta, action_set, nb_actions, d = load_amazon(reviews_x_items, size_catalog=1000, seed=seed)

    return theta, action_set, nb_actions, d

