import joblib
import pandas as pd

from find_your_inner_gamer.gcp import get_model_from_gcp, get_neighbors_from_gcp
from find_your_inner_gamer.utils import get_img
from find_your_inner_gamer.data import get_local_data


def download_model_local():
    return joblib.load('model.joblib')

def predict(game, df):
    """Finds a list of similar games.

    Args:
        game (name): Name of the game teh user want.
        df (pandas.DataFrame): Dataframe for the games

    Returns:
        pandas.DataFrame: Dataframe with recommendations.
    """
    try:
        model = download_model_local()
    except:
        model = get_model_from_gcp()

    X_neighbors = get_neighbors_from_gcp()
    X_neighbors.set_index('Unnamed: 0', drop=True, inplace=True)

    neighbors_index = model.kneighbors(X_neighbors.loc[[game]],n_neighbors=10)[1][0]

    new_df_values = {
        'title' : [],
        'url': [],
        'price': [],
        'reviews': [],
        'op_sys': [],
        'developer': [],
        'image_url' : []
    }

    for index in neighbors_index:
        new_df_values['title'].append(df.loc[index, 'name'])
        new_df_values['url'].append(df.loc[index, 'url'])
        new_df_values['price'].append(df.loc[index, 'price'])
        new_df_values['reviews'].append(df.loc[index, 'reviews'])
        new_df_values['op_sys'].append(df.loc[index, 'op_sys'])
        new_df_values['developer'].append(df.loc[index, 'developer'])
        new_df_values['image_url'].append(get_img(df.loc[index, 'url']))

    recommendations = pd.DataFrame(data=new_df_values)
    recommendations.fillna('no value', inplace=True)
    return recommendations

def print_games(games, start):
    """
    Displays the games in a menu format

    Args:
        games (list): List of games
        start (int): Index starting value
    """
    for index, game in enumerate(games, start=start):
        print(f"{index} - {game}")


def get_game(df):
    """
    Find the game the user wants from the dataframe

    Args:
        df (Dataframe): Games dataframe

    Returns:
        str: Name of the game choosen by the user
    """

    name = input("What game are you looking for?\n>") # ask the user for a game

    # checks how many possible games exist in the dataframe
    game_options = list(
        df[df['name'].\
               str.contains(name, case=False) == True]['name']
    )

    # Game don't exists in the data frame
    if len(game_options) == 0:
        print('Not Found')
        return None

    # If there is only one game
    if len(game_options) == 1:
        return game_options[0]

    # Less then 10 games only one page
    if len(game_options) <= 10:
        print_games(game_options, 0)
        option = input('Choose the number for the game you want:\n> ')
        return game_options[int(option)]

    # Loops the games 10 by 10
    page_num = int(len(game_options)/10) # number of iterations to show games
    last_index = 11
    prev_index = 0

    for page in range(page_num+1):
        print_games(game_options[prev_index:last_index], prev_index)

        # after each iterations ask the user if the game is in that page
        option = input('Choose the number for the game you want(-1 if not shown):\n> ')
        if prev_index <= int(option) <= last_index:
            return game_options[int(option)]

        # if it has shown all pages it means it was not in the dataset
        if last_index >= page_num * 10:
            return 'Not Found'

        # updates the index for the next page
        prev_index = last_index
        last_index += 10

if __name__ == "__main__":
    df = get_local_data()
    game = get_game(df)
    print(predict(game, df))
