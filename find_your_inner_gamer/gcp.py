import os
import pandas as pd
import joblib

from google.cloud import storage
from find_your_inner_gamer.params import BUCKET_NAME, BUCKET_CSV_DATA_PATH,MODEL_NAME, MODEL_VERSION


def get_data_from_gcp():
    """Function to get the data from the cloud

    Returns:
        pandas.DataFrame: dataframe containing info about games
    """
    path = f"gs://{BUCKET_NAME}/{BUCKET_CSV_DATA_PATH}"
    return pd.read_csv(path)


def get_neighbors_from_gcp():
    """Function to get the data about output of the preprocessing from the cloud

    Returns:
        pandas.DataFrame: dataframe containing the preprocessing data
    """
    path = f"gs://{BUCKET_NAME}/data/X_neighbors.csv"
    return pd.read_csv(path)


def get_model_from_gcp():
    """Function to get the trained model from teh cloud

    Returns:
        joblib: Trained model
    """
    client = storage.Client().bucket(BUCKET_NAME)

    local_model_name = 'model.joblib'
    model_storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(model_storage_location)
    blob.download_to_filename('model.joblib')
    return joblib.load('model.joblib')


def storage_upload(rm=True):
    """Function to save model & preprocessing outcome to the cloud

    Args:
        rm (bool, optional): Especifies if the files should be removed from local folder. Defaults to True.
    """
    client = storage.Client().bucket(BUCKET_NAME)

    local_model_name = 'model.joblib'
    model_storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(model_storage_location)
    blob.upload_from_filename('model.joblib')
    print(f"=> model.joblib uploaded to bucket {BUCKET_NAME} inside {model_storage_location}")

    local_neighbors_csv = 'X_neighbors.csv'
    neighbors_storage_location = f"data/{local_neighbors_csv}"
    blob = client.blob(neighbors_storage_location)
    blob.upload_from_filename('X_neighbors.csv')
    print(f"=> X_neighbors.csv uploaded to bucket {BUCKET_NAME} inside {neighbors_storage_location}")

    if rm:
        os.remove('model.joblib')
        os.remove('X_neighbors.csv')
