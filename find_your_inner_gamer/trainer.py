import pandas as pd
import numpy as np
import joblib
# Preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
# Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from find_your_inner_gamer.utils import kmeans_labels
from find_your_inner_gamer.data import get_local_data
from find_your_inner_gamer.gcp  import storage_upload, get_data_from_gcp

class Trainer(object):
    def __init__(self, X, y):
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        # deals with size issue from vectorizing features
        array_transf = FunctionTransformer(lambda array: array.toarray())
        # adds the feature cluster to the df
        self.X['cluster'] = kmeans_labels(self.X)

        # preprocessing of metadata
        meta_transf = make_pipeline(
            TfidfVectorizer(min_df=0.03),
            array_transf,
            RobustScaler()
        )

        # creates an order for the different reviews
        ord_encoder = OrdinalEncoder(
            categories=[
                [
                    "Overwhelmingly Negative",
                    "Very Negative",
                    "Negative",
                    "Mostly Negative",
                    'Mixed',
                    "Mostly Positive",
                    "Positive",
                    "Very Positive",
                    "Overwhelmingly Positive"
                ]],
            dtype=np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        # preprocessing of reviews
        ord_transf = make_pipeline(
            ord_encoder,
            StandardScaler()
        )

        # preprocessing of clusters
        cluster_transf = make_pipeline(
            OneHotEncoder(sparse=False),
            StandardScaler()
        )

        #preprocessing for numerical features
        num_transf = make_pipeline(StandardScaler())

        # creates a pipeline to run the different preprocessing tranformers in parallel
        preproc_basic = make_column_transformer(
            (meta_transf, 'metadata'),
            (cluster_transf, ['cluster']),
            (ord_transf, ['reviews']),
            (num_transf, ['mature_content', 'achievements']),
            remainder='drop'
        )

        # fitting PCA transformation for the previous pipeline
        full_pipe = make_pipeline(preproc_basic, PCA(n_components=10) )
        self.pipeline  = full_pipe.fit_transform(self.X)


    def train(self):
        """Trains the model with KNeighbors using the preprocessing pipeline.

        Returns:
            KNeighborsRegressor: The fitted k-nearest neighbors regressor
        """
        self.set_pipeline()
        X_neighbors = pd.DataFrame(self.pipeline, index=self.X.name.tolist())
        X_neighbors.to_csv('X_neighbors.csv')
        return KNeighborsRegressor().fit(X_neighbors, self.y)

    def save_model(self, model):
        """Saves the model and the uploads it to the cloud.

        Args:
            model (joblib): Trained model.
        """
        joblib.dump(model, 'model.joblib')
        print("model.joblib saved locally")
        storage_upload()


if __name__ == "__main__":
    df = get_local_data() # gets data locally
    #df = get_data_from_gcp() # gets data from the cloud

    trainer = Trainer(df, df['url'])
    model = trainer.train()
    trainer.save_model(model)
