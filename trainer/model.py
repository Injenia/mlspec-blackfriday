from typing import List
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs


class UserModel(tf.keras.Model):

    def __init__(self,
                 user_features: dict,
                 embedding_dim: int = 32,
                 layer_units:List[int] = [],
                 user_input_embedding_dim: int = 32,
                 user_input_embedding_l1:float = 0.0,
                 user_input_embedding_l2:float = 0.0
                ):
        """
        Create a user embedding on features provided, each feature should specify all
        the unique values in order to build the lookup layer.

        :param user_features: dictionary with <feature_name> : <feature_unique_values>
        :param embedding_dim: dimension of the embedding
        """
        super().__init__()

        self.user_features = {}
        for feature_name, unique_values in user_features.items():
            feature_layer = tf.keras.Sequential(
                [
                    tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_values, mask_token=None),
                    tf.keras.layers.Embedding(
                        len(unique_values) + 1, 
                        user_input_embedding_dim, 
                        embeddings_regularizer=tf.keras.regularizers.L1L2(l1=user_input_embedding_l1, l2=user_input_embedding_l2)
                    ),
                ],
                name=feature_name)
            self.user_features[feature_name] = feature_layer
          
        self.net_body = None
        if len(layer_units) > 0:
            self.net_body = tf.keras.Sequential([
                tf.keras.layers.Dense(units, activation='relu')
                for units in layer_units
            ])
            

    def call(self, inputs, **kwargs):
        layers_stack = []
        feature_names=sorted(self.user_features.keys())

        for feature_name in feature_names:
            feature_layer = self.user_features[feature_name]
            layer_valorized = feature_layer(inputs[feature_name])
            layers_stack.append(layer_valorized)

        embeddings = tf.concat(layers_stack, axis=1)
        return embeddings if self.net_body is None else self.net_body(embeddings)

    def get_config(self):
        raise NotImplementedError


class ProductModel(tf.keras.Model):

    def __init__(self,
                 product_features: dict,
                 embedding_dim: int = 32,
                 layer_units:List[int] = [],
                 product_input_embedding_dim:int = 32,
                 product_input_embedding_l1:float = 0.0,
                 product_input_embedding_l2:float = 0.0
                ):
        """
        Create a product embedding model.
        Require the list of unique products ids in order to create the lookup layer

        :param product_features: unique product features used to calculate the embedding
        :param embedding_dim: dimension of the embedding
        """
        super().__init__()
        
        self.product_features = {}
        for feature_name, unique_values in product_features.items():
            feature_layer = tf.keras.Sequential(
                [
                    tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_values, mask_token=None),
                    tf.keras.layers.Embedding(
                        len(unique_values) + 1, 
                        product_input_embedding_dim,
                        embeddings_regularizer=tf.keras.regularizers.L1L2(l1=product_input_embedding_l1, l2=product_input_embedding_l2)
                    )
                ])
            self.product_features[feature_name]=feature_layer
            
        self.net_body = None
        if len(layer_units) > 0:
            self.net_body = tf.keras.Sequential([
                tf.keras.layers.Dense(units, activation='relu')
                for units in layer_units
            ])
            

    def call(self, inputs, **kwargs):
        #embedding = tf.concat([self.product_embedding(products_id)], axis=1)
        #return embedding
        layers_stack = []
        feature_names=sorted(self.product_features.keys())

        for feature_name in feature_names:
            feature_layer = self.product_features[feature_name]
            layer_valorized = feature_layer(inputs[feature_name])
            layers_stack.append(layer_valorized)

        embeddings = tf.concat(layers_stack, axis=1)
        return embeddings if self.net_body is None else self.net_body(embeddings)
        

    def get_config(self):
        raise NotImplementedError


class BlackFridayModel(tfrs.models.Model):

    def __init__(self,
                 user_model,
                 product_model,
                 user_features: dict,
                 product_features: dict, #product_unique_ids: np.ndarray,
                 topk_candidates: tf.data.Dataset,
                 embedding_dim: int = 32,
                 topk_metric_batch_size: int = 128,
                 product_id_col_name: str = "Product_ID",
                 user_layers: List[int] = [],
                 product_layers: List[int] = [],
                 user_input_embedding_dim:int = 32,
                 user_input_embedding_l1:float = 0.0,
                 user_input_embedding_l2:float = 0.0,
                 product_input_embedding_dim:int = 32,
                 product_input_embedding_l1:float = 0.0,
                 product_input_embedding_l2:float = 0.0,
                 temperature = 1.0
                 ):
        """
        Utility class used to train the two embedding models.

        :param user_model: user keras model
        :param product_model: product keras model
        :param topk_candidates: dataset with candidates to evaluate topk metric
        :param user_features: unique user features used to calculate the embedding
        :param product_features: unique product features used to calculate the embedding
        :param embedding_dim: the embedding dimension
        :param topk_metric_batch_size: topk metric dataset batch size
        :param product_id_col_name: name of the column on the dataset with product identifier
        :param user_layers: list of dimensions of layers in the user model
        :param product_layers: list of dimensions of layers in the product model
        """
        super().__init__()
        self.product_embedder = product_model
        self.user_embedder = user_model
        self.user_features = user_features.keys()
        self.product_features = product_features.keys()
        self.embedding_dim = embedding_dim
        self.product_id_col_name = product_id_col_name
        # `User` is the query entity
        self.query_model = tf.keras.Sequential([
            user_model(user_features, embedding_dim, user_layers, user_input_embedding_dim, user_input_embedding_l1, user_input_embedding_l2),
            tf.keras.layers.Dense(embedding_dim)
        ])

        # `Products` are the candidates entity
        self.candidate_model = tf.keras.Sequential([
            product_model(product_features, embedding_dim, product_layers, product_input_embedding_dim, product_input_embedding_l1, product_input_embedding_l2),
            tf.keras.layers.Dense(embedding_dim)
        ])

        # See https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval
        #topk_candidates_embedded = topk_candidates.batch(topk_metric_batch_size).map(self.candidate_model)
        topk_candidates_embedded = topk_candidates.batch(topk_metric_batch_size).map(self.candidate_model)
        self.task = tfrs.tasks.Retrieval(
            temperature = temperature,
            metrics=tfrs.metrics.FactorizedTopK(candidates=topk_candidates_embedded),
        )

    def get_user_tower(self):
        return self.query_model

    def get_product_tower(self):
        return self.candidate_model

    def compute_loss(self, features, training=False):
        query_data = {feature_name: features[feature_name] for feature_name in self.user_features}
        product_data =  {feature_name: features[feature_name] for feature_name in self.product_features}
        query_embeddings = self.query_model(query_data)
        #product_embeddings = self.candidate_model(features[self.product_id_col_name])
        product_embeddings = self.candidate_model(product_data)

        # Retrieval call: https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval
        # "The task will try to maximize the affinity of these query, candidate pairs while minimizing
        # the affinity between the query and candidates belonging to other queries in the batch."
        return self.task(query_embeddings=query_embeddings,
                         candidate_embeddings=product_embeddings,
                         compute_metrics=True, #training,  # disable during training for better performances
                         candidate_ids=None,
                         )

    def get_config(self):
        raise NotImplementedError()

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError()


def create_bf_model(user_features: dict,
                    product_features, #product_unique_ids,
                    embedding_dim,
                    learning_rate,
                    topk_candidates,
                    topk_metric_batch_size,
                    user_layers,
                    product_layers,
                    user_input_embedding_dim,
                    user_input_embedding_l1,
                    user_input_embedding_l2,
                    product_input_embedding_dim,
                    product_input_embedding_l1,
                    product_input_embedding_l2,
                    temperature
                    ):
    model = BlackFridayModel(UserModel, ProductModel,
                             user_features,
                             product_features, #product_unique_ids,
                             topk_candidates,
                             embedding_dim,
                             topk_metric_batch_size,
                             user_layers=user_layers,
                             product_layers=product_layers,
                             user_input_embedding_dim=user_input_embedding_dim,
                             user_input_embedding_l1=user_input_embedding_l1,
                             user_input_embedding_l2=user_input_embedding_l2,
                             product_input_embedding_dim=product_input_embedding_dim,
                             product_input_embedding_l1=product_input_embedding_l1,
                             product_input_embedding_l2=product_input_embedding_l2,
                             temperature = temperature
                             )

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate))
    return model
