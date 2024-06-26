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
        path += 'movielens_25m/data/movielens.zip'
        if init:
            data = pd.read_csv(path, sep=",", index_col=0)
            return [data]
        else:
            df_data = df_data[0]
            ### One hot encoding of users
            action_set, nb_actions, d = get_movies(df_data, seed)

    elif name == 'Steam':
        path += 'steam/data'
        if init:
            recommendations = pd.read_csv(f'{path}/steam_links.zip')
            games_vectors = pd.read_csv(f'{path}/steam_actions.zip')
            return [recommendations, games_vectors]
        else:
            recommendations, games_vectors = df_data
            theta, action_set, nb_actions, d = load_steam(recommendations, games_vectors, size_catalog=nb_actions, seed=seed)

    elif name == 'Yahoo':
        path += 'yahoo/data/yahoo.zip'
        if init:
           reviews_x_items = pd.read_csv(path, sep=",", index_col=0)
           return [reviews_x_items]
        else:
            reviews_x_items = df_data[0]
            theta, action_set, nb_actions, d = load_yahoo(reviews_x_items, seed=seed)

    elif name == 'Amazon':
        path += 'amazon/data/amazon.zip'
        if init:
           reviews_x_items = pd.read_csv(path, sep=",", index_col=0)
           return [reviews_x_items]
        else:
            reviews_x_items = df_data[0]
            theta, action_set, nb_actions, d = load_amazon(reviews_x_items, size_catalog=1000, seed=seed)

    return theta, action_set, nb_actions, d

