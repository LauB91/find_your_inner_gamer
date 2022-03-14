from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from find_your_inner_gamer.gcp import get_data_from_gcp, get_model_from_gcp, get_neighbors_from_gcp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    """Home page for API

    Returns:
        dic: Dictonary just saying 'hello word'
    """
    return {"greeting": "Hello world"}


@app.get("/predict")
def predict(game):
    """Recommendation part of the API

    Args:
        game (str): Name of the game the user wants likes

    Returns:
        dic: Dictonary with a list of game titles that are similar to the one the user inputed
    """
    df = get_data_from_gcp()
    df = df.fillna('no value')
    model = get_model_from_gcp()
    X_neighbors = get_neighbors_from_gcp()
    X_neighbors.set_index('Unnamed: 0',drop=True, inplace = True)
    neighbors_index = model.kneighbors(X_neighbors.loc[[game]],n_neighbors=10)[1][0]

    new_df_values = {
         'title' : [df.loc[index, 'name'] for index in neighbors_index],
     }

    return new_df_values
