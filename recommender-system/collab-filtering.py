import numpy as np
import pandas
import matplotlib.pyplot as plt
from math import sqrt


def read_data():
    movies = pandas.read_csv("movies.csv")
    ratings = pandas.read_csv("ratings.csv")

    movies['year'] = movies.title.str.extract("(\(\d\d\d\d\))", expand=False)
    movies['year'] = movies.year.str.extract('(\d\d\d\d)', expand=False)
    movies['title'] = movies.title.str.replace('(\(\d\d\d\d\))', '')
    movies['title'] = movies['title'].apply(lambda x: x.strip())

    movies = movies.drop('genres', 1)
    # print(movies.head())

    ratings = ratings.drop('timestamp', 1)
    return movies, ratings


def user_dataframe(movies):
    user_input = [
        {'title': 'Breakfast Club, The', 'rating': 5},
        {'title': 'Toy Story', 'rating': 3.5},
        {'title': 'Jumanji', 'rating': 2},
        {'title': "Pulp Fiction", 'rating': 5},
        {'title': 'Akira', 'rating': 4.5}
    ]
    input_movies = pandas.DataFrame(user_input)

    input_id = movies[movies['title'].isin(input_movies['title'].tolist())]
    input_movies = pandas.merge(input_id, input_movies)
    input_movies = input_movies.drop('year', 1)
    return input_movies


def rate_users(user_input, ratings, movies):
    # select users who gave ratings to the same movies as the user
    user_subset = ratings[ratings['movieId'].isin(user_input['movieId'].tolist())]
    # group by the userId, creates subdataframes each with the same userId
    user_subset_grouped = user_subset.groupby(['userId'])

    # sort so that users that have most common with have higher priority
    user_subset_grouped = sorted(user_subset_grouped, key=lambda x: len(x[1]), reverse=True)

    user_subset_grouped = user_subset_grouped[0:100]

    pearson_corr_dict = {}

    for name, group in user_subset_grouped:
        group = group.sort_values(by='movieId')
        user_input = user_input.sort_values(by='movieId')

        num_ratings = len(group)
        temp_df = user_input[user_input['movieId'].isin(group['movieId'].tolist())]
        temp_rating_list = temp_df['rating'].tolist()
        temp_group_rating_list = group['rating'].tolist()

        Sxx = sum([i*i for i in temp_rating_list]) - pow(sum(temp_rating_list), 2) / num_ratings
        Syy = sum([i*i for i in temp_group_rating_list]) - pow(sum(temp_group_rating_list), 2) / num_ratings
        Sxy = sum([i*j for i, j in zip(temp_rating_list, temp_group_rating_list)]) - \
              sum(temp_rating_list) * sum(temp_group_rating_list) / num_ratings

        if Sxx!=0 and Syy!=0:
            pearson_corr_dict[name] = Sxy / (Sxx * Syy)
        else:
            pearson_corr_dict[name] = 0

    pearson_df = pandas.DataFrame.from_dict(pearson_corr_dict, orient='index')
    pearson_df.columns = ['similarityIndex']
    pearson_df['userId'] = pearson_df.index
    pearson_df.index = range(len(pearson_df))

    top_users = pearson_df.sort_values(by='similarityIndex', ascending=False)[0:50]

    return top_users


def recommender(top_users, ratings):
    top_users_rating = top_users.merge(ratings, left_on='userId', right_on='userId', how='inner')
    top_users_rating['weightedRating'] = top_users_rating['rating']*top_users_rating['similarityIndex']
    temp_top_users_rating = top_users_rating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
    temp_top_users_rating.columns = ['sum_similarityIndex', 'sum_weightedRating']

    recommendation_df = pandas.DataFrame()
    recommendation_df['weighted average recommendation score'] = temp_top_users_rating['sum_weightedRating']/temp_top_users_rating['sum_similarityIndex']
    recommendation_df['movieId'] = temp_top_users_rating.index
    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
    return recommendation_df


if __name__ == '__main__':
    movies, ratings = read_data()
    user_df = user_dataframe(movies)
    top_users = rate_users(user_df, ratings, movies)
    rec_df = recommender(top_users, ratings)
    final_recommendation = movies.loc[movies['movieId'].isin(rec_df.head(10)['movieId'].tolist())]
    print(final_recommendation)