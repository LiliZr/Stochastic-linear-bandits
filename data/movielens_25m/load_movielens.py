import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, PolynomialFeatures

FILES = ["links", "movies", "ratings", "tags"]

def load_movielens(path):
    # Load all data in a dict
    df_data = {}
    for filename in FILES:
        df_data[filename] = pd.read_csv(f'{path}/{filename}.csv', sep=",")
    return df_data

def get_year(title):
    idx = title.rfind('(') + 1
    try:
        return int(title[idx:idx+4])
    except ValueError:
        print(title, end='\t')
        return -1
    
def extract_features(df_data, id_column):
    features = None
    movies = df_data['movies'].copy()
    ratings = df_data['ratings'].copy().drop(['timestamp'], axis=1)
    name = None
    if id_column == 'movieId':
        name = 'movie_'
        features = movies.copy()
    
        # Create year column
        features['year'] = features['title'].map(get_year)

        # Drop title
        features.drop('title', inplace=True, axis=1)

        # Encode genres
        features['genres'] = features['genres'].apply(lambda x: x.split('|'))
        mlb = MultiLabelBinarizer()
        features = features.join(pd.DataFrame(mlb.fit_transform(features.pop('genres')),
                                columns=mlb.classes_,
                                index=features.index))
    else:
        name = ''
        features = ratings.copy()

    # Stats ratings 
    stats = {}
    aggregates = ['mean', 'median', 'min', 'max', 'std', 'size']

    for agg in  aggregates:
        name_agg = 'number' if agg == 'size' else agg
        stats[agg] = ratings.groupby(id_column)['rating'].aggregate(agg)
        stats[agg] = pd.DataFrame({id_column: stats[agg].index, f'{name}{name_agg}_ratings': stats[agg].values})

    # Merging all dataframes
    for rating_stat in stats.values():
        features = pd.merge(features, rating_stat, on=id_column)
    
    # Replace NaN with 0
    features.fillna(0, inplace=True)

    # Add column index
    features[f'{name}index'] =  features.index

    return features



def get_features_users_x_movies(df_data,  nb_movies = 500):
    """
        Create features for users and movies: Merge users and movies by 
            concatenating features of users with features of movies they have seen
        param(s):
            nb_movies: threshold of movies rated by a user to keep it
        return 
            (tuple): (concatenation features user x movies, user ids)

    """
    # Compute movies features
    movies = extract_features(df_data, 'movieId')
    # Compute some users features
    ratings = extract_features(df_data, 'userId')

    # Keep only users that rated more than nb_movies movies
    ratings = ratings[ratings['number_ratings'] > nb_movies]
    # Merge ratings x movies
    ratings_x_movies = pd.merge(ratings, movies, on="movieId")
    # Compute new users features from old features and aggregation of features of movies seen 
    users_features = ratings_x_movies.groupby('userId').aggregate('mean')
    users_features.drop(['movieId', 'rating', 'movie_index'], axis=1, inplace=True)
    users_features = users_features.add_prefix('user_')
    # Merge users and movies: concatenate features of users with features of movies they have seen
    users_x_movies = pd.merge(users_features, ratings_x_movies, on='userId')
    return users_x_movies

def get_movies_ratings_user(users_x_movies, seed=12):
    rng = np.random.RandomState(seed)
    users_ids = users_x_movies['userId'].unique()

    # Choose user
    user_id = rng.choice(users_ids)
    # print(f'Chosen User id: {user_id}')

    # Extract context vectors of corresponding userId (list of features user concatenated with features movies) 
    movies_userId = users_x_movies[users_x_movies['userId'] == user_id]

    # Keep rewards and movie ids
    rewards = np.array(movies_userId['rating']/5)
    movie_ids = np.array(movies_userId['movieId'])

    # Drop useless features
    columns_to_drop = ['userId', 'rating', 'user_index', 'index', 'movie_index', 'movieId', 
                        'mean_ratings', 'median_ratings', 'min_ratings',
                        'max_ratings', 'std_ratings', 'number_ratings', ]
    movies_userId = movies_userId.drop(columns_to_drop, axis=1)
    movies_userId = np.array(movies_userId)
    
    # Create feature map from existing features user x movies
    poly = PolynomialFeatures(2, interaction_only=True)
    movies_userId_fm = poly.fit_transform(movies_userId)

    # Normalize
    movies_userId_fm/=np.linalg.norm(movies_userId_fm, 2, axis=1)[:, None]
    return (movies_userId_fm, rewards), movies_userId_fm.shape[0], movies_userId_fm.shape[1]
    



def get_movies(ratings_x_movies, seed=12):
    rng = np.random.RandomState(seed)
    users_ids = ratings_x_movies['userId'].unique()

    # Choose user
    user_id = rng.choice(users_ids)

    # Group by movieId to get list of users that rated the movie
    ratings_x_movies.groupby('movieId')
    user_ids = ratings_x_movies.groupby('movieId')['userId'].apply(list)
    ratings_x_movies = pd.merge(user_ids, ratings_x_movies, on="movieId")

    # Get movies watched by chosen user
    movies = ratings_x_movies[ratings_x_movies['userId_y'] == user_id]

    # Keep rewards and movie ids
    rewards = np.array(movies['rating']/5)
    movie_ids = np.array(movies['movieId'])

    # Users to "one-hot"
    mlb = MultiLabelBinarizer()
    mlb.fit(user_ids)
    movies = movies.join(pd.DataFrame(mlb.transform(movies.pop('userId_x')),
                                                columns=mlb.classes_,
                                                index=movies.index))
    # Drop useless features
    columns_to_drop = ['userId_y', 'rating', 'movieId', 'timestamp', 'movie_index']
    movies = movies.drop(columns_to_drop, axis=1)

    # Normalize
    movies = np.array(movies)
    movies /= np.linalg.norm(movies, 2, axis=1)[:, None]
    
    
    return (movies, rewards), movies.shape[0], movies.shape[1]