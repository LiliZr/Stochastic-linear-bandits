import numpy as np 
from sklearn.linear_model import LinearRegression

def load_yahoo(users_x_items,  seed=12):
    items_vectors = users_x_items.drop(['click', 'user'], axis=1, inplace=False)
    items_vectors.drop_duplicates(inplace=True)

    nb_items_total = len(items_vectors)     # total nb items
    nb_items_lr = 50                        # nb items for linear regression

    ### Get users listp
    rng = np.random.RandomState(seed)
    users = users_x_items['user'].unique()

    ### Choose user
    user_id = rng.choice(users)

    ### Get items already reviewed by user
    items_rated = users_x_items[users_x_items['user'] == user_id]
    items_rated_ids = items_rated['item_id'].values[:nb_items_lr]
    items_ratings = items_rated['click'].values[:nb_items_lr]
    items_rated = np.array(items_rated.drop(['item_id', 'user', 'click',], axis=1, inplace=False))[:nb_items_lr]
    items_rated /= np.linalg.norm(items_rated, 2, axis=1)[:, None]

    ### Compute linear regression to estimate theta 
    reg = LinearRegression(fit_intercept=False).fit(items_rated, items_ratings)
    theta_user = reg.coef_

    ### Create catalog of X items not rated by given user using most rated games
    size_catalog = nb_items_total - nb_items_lr
    items_catalog = items_vectors[~items_vectors['item_id'].isin(items_rated_ids)]
    items_catalog = np.array(items_catalog[:size_catalog].drop('item_id', axis=1))


    # theta_user /= np.linalg.norm(theta_user, 2)
    items_catalog /= np.linalg.norm(items_catalog, 2, axis=1)[:, None]

    return theta_user, items_catalog, size_catalog, theta_user.shape[0]