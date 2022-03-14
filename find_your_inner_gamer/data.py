import pandas as pd
from find_your_inner_gamer.params import LOCAL_DATA_PATH

def get_local_data():
    """Fucntion to get the data locally

    Returns:
        pandas.DataFrame: Dataframe for our games.
    """
    return pd.read_csv(LOCAL_DATA_PATH)
