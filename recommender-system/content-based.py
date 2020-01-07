import numpy as np
import pandas
import matplotlib.pyplot as plt
from math import sqrt


def read_data():
    movies = pandas.read_csv("movies.csv")
    ratings = pandas.read_csv("ratings.csv")

    # create a new column for the movie years by extracting them from the titles
    movies['year'] = movies.title.str.extract('(\(\d\d\d\d\))', expand=False)
    # remove the parentheses from the years
    movies['year'] = movies.year.str.extract('(\d\d\d\d)', expand=False)
    # remove the years from the titles
    movies['title'] = movies.title.str.replace('(\(\d\d\d\d\))', '')
    # remove any ending whitespace
    movies['title'] = movies['title'].apply(lambda x: x.strip())

    # split the genres
    movies['genres'] = movies.genres.str.split('|')

    # copy movies dataframe to a new one that will have genres as one-hot vectors
    movies_genres = movies.copy()
    for index, row in movies_genres.iterrows():
        for genre in row['genres']:
            movies_genres.at[index, genre] = 1

    movies_genres = movies_genres.fillna(0)
    movies_genres = movies_genres.drop('genres', 1)
    #print(movies_genres.head())

    # drop the timestamp column
    ratings = ratings.drop('timestamp', 1)
    #print(ratings.head())

    return movies, movies_genres, ratings


def user_dataframe(movies):
    user_input = [
        {'title': 'Breakfast Club, The', 'rating': 5},
        {'title': 'Toy Story', 'rating': 3.5},
        {'title': 'Jumanji', 'rating': 2},
        {'title': "Pulp Fiction", 'rating': 5},
        {'title': 'Akira', 'rating': 4.5}
    ]
    # create dataframe from the user input
    input_df = pandas.DataFrame(user_input)

    # find the movie id in the movies dataframe and merge
    input_id = movies[movies['title'].isin(input_df['title'].tolist())]

    # merge input movies with their ids and drop unnecessary columns
    input_movies = pandas.merge(input_id, input_df)
    input_movies = input_movies.drop('genres', 1).drop('year', 1)
    # print(input_movies.head())

    return input_movies


def recommender(input_movies, movies_genres):
    user_movies = movies_genres[movies_genres['movieId'].isin(input_movies['movieId'].tolist())]
    # print(user_movies)

    user_movies = user_movies.reset_index(drop=True)
    user_genre = user_movies.drop('movieId', 1).drop('title', 1).drop('year', 1)
    # print(user_genre.columns)
    # print(user_genre.dtypes)

    # print(input_movies['rating'])
    user_profile = user_genre.transpose().dot(input_movies.rating)
    # print(user_profile)

    genre_table = movies_genres.set_index(movies_genres['movieId'])
    genre_table = genre_table.drop('movieId', 1).drop('title', 1).drop('year', 1)

    recommendation_table = (genre_table*user_profile).sum(axis=1)/user_profile.sum()
    recommendation_table = recommendation_table.sort_values(ascending=False)
    # print(recommendation_table)

    return recommendation_table


if __name__ == '__main__':
    movies, movies_genres, ratings = read_data()
    user_input = user_dataframe(movies)
    rec_table = recommender(user_input, movies_genres)

    print(movies.loc[movies['movieId'].isin(rec_table.head(20).keys())])
