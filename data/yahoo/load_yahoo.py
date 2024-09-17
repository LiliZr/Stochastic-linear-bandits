import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  OneHotEncoder


def load_yahoo(users_x_items,  seed=12):
    # Get users
    rng = np.random.RandomState(seed)
    users_ids = users_x_items['user'].unique()

    contexts_arms_rewards = []

    limit_users = 10
    # For each user
    for i, user_id in enumerate(users_ids[:limit_users]):
        # Convert user to vector
        u = user_id.split(' ')
        u = np.array(list(map(lambda x: float(x[2:]), u)))
        context = u

        # Get items already reviewed by user
        items_rated = users_x_items[users_x_items['user'] == user_id]
        items_rated_ids = items_rated['item_id'].values
        items_ratings = items_rated['click'].values
        items_ratings[items_ratings > 0] = 1
        items_rated = np.array(items_rated.drop(['item_id', 'user', 'click',], axis=1, inplace=False))
        items_rated /= np.linalg.norm(items_rated, 2, axis=1)[:, None]


        context_r = np.tile(context, (items_rated.shape[0], 1))
        contexts_arms_rewards.append((np.concatenate((context_r, items_rated), axis=1), items_ratings))
        
    return contexts_arms_rewards, items_rated.shape[0], items_rated.shape[1]