import numpy as np 
from sklearn.linear_model import LinearRegression



def load_steam(recommendations, games_vectors, size_catalog=2000, seed=12):
    rng = np.random.RandomState(seed)
    users = recommendations['user_id'].unique()

    ### Choose user
    user_id = rng.choice(users)

    ### Get games already  recommended (1) or not (0) by user
    games_recommended = recommendations[recommendations['user_id'] == user_id]
    games_recommended_ids = games_recommended['app_id'].values
    games_recommendations = games_recommended['is_recommended'].values
    games_recommended = games_vectors[games_vectors['app_id'].isin(games_recommended_ids)]
    games_recommended = np.array(games_recommended.drop('app_id', axis=1, inplace=False))
    games_recommended /= np.linalg.norm(games_recommended, 2, axis=1)[:, None]

    ### Compute linear regression to estimate theta of an estimator
    reg = LinearRegression(fit_intercept=False).fit(games_recommended, games_recommendations)
    theta_user = reg.coef_

    ### Create catalog of 2000 games not played by given user using most rated games
    games_catalog = games_vectors[~games_vectors['app_id'].isin(games_recommended_ids)]
    games_catalog = np.array(games_catalog[:size_catalog].drop('app_id', axis=1))


    # theta_user /= np.linalg.norm(theta_user, 2)
    games_catalog /= np.linalg.norm(games_catalog, 2, axis=1)[:, None]

    return theta_user, games_catalog, size_catalog, theta_user.shape[0]

