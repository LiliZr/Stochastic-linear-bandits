import numpy as np 
from sklearn.linear_model import LinearRegression


def load_amazon(reviews_x_items, size_catalog=1500, seed=12):
    items_vectors = reviews_x_items.drop(['rating', 'user_id', 'verified_purchase'], axis=1, inplace=False)
    items_vectors.drop_duplicates(inplace=True)

    ### Get users
    rng = np.random.RandomState(seed)
    users = reviews_x_items['user_id'].unique()

    ### Choose user
    user_id = rng.choice(users)

    ### Get items already  reviewed by user
    items_rated = reviews_x_items[reviews_x_items['user_id'] == user_id]
    items_rated_ids = items_rated['parent_asin'].values
    items_ratings = items_rated['rating'].values
    items_rated = np.array(items_rated.drop(['parent_asin', 'rating', 'user_id', 'verified_purchase'], axis=1, inplace=False))
    items_rated /= np.linalg.norm(items_rated, 2, axis=1)[:, None]

    ### Compute linear regression to estimate theta 
    reg = LinearRegression(fit_intercept=False).fit(items_rated, items_ratings)
    theta_user = reg.coef_

    ### Create catalog of X items not rated by given user using most rated games
    items_catalog = items_vectors[~items_vectors['parent_asin'].isin(items_rated_ids)]
    items_catalog = items_catalog.sort_values(by='item_number_ratings', ascending=False, inplace=False)
    items_catalog = np.array(items_catalog[:size_catalog].drop('parent_asin', axis=1))


    # theta_user /= np.linalg.norm(theta_user, 2)
    items_catalog /= np.linalg.norm(items_catalog, 2, axis=1)[:, None]

    return theta_user, items_catalog, size_catalog, theta_user.shape[0]
