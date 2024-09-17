import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


def load_steam(recommendations, games_vectors, size_catalog=2000, seed=12):
    rng = np.random.RandomState(seed)
    users = recommendations['user_id'].unique()

    print(len(users))
    # Encode users(contexts) using one hot encoding
    limit_users = 10
    encoder = OneHotEncoder(sparse_output=False)
    contexts = encoder.fit_transform(users[:limit_users].reshape(-1, 1))

    contexts_arms_rewards = []

    # For each user
    for i, user_id in enumerate(users[:limit_users]):
        context = contexts[i]
        # Items rated by user
        rated_ids = list(recommendations[recommendations['user_id'] == user_id]['app_id'].unique())


        # Create catalog of items from recommended and non-recommended items
        weights = np.array((games_vectors['app_id'].isin(rated_ids)).astype(float))
        prob = 50
        weights[weights == 1] = prob
        weights[weights == 0] = 100 - prob
        games = games_vectors.sample(n=size_catalog, random_state=seed, weights=weights)

        # Keep rewards and items ids
        rewards = np.array((games['app_id'].isin(rated_ids)).astype(int))
        games_ids = np.array(games['app_id'])


        # Drop useless features
        columns_to_drop = ['app_id', ]
        games = games.drop(columns_to_drop, axis=1)

        # Normalize
        games = np.array(games)
        games /= np.linalg.norm(games, 2, axis=1)[:, None]

        context_r = np.tile(context, (games.shape[0], 1))
        contexts_arms_rewards.append((np.concatenate((context_r, games), axis=1), rewards))

    return contexts_arms_rewards, games.shape[0], games.shape[1]

