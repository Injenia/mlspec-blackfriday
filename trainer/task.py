import argparse
import json
import logging
import os
from typing import List

import tensorflow as tf
import tensorflow_recommenders as tfrs

from . import model
from . import util


examples_by_feature = {  # used to check scann consistency
        "Gender": tf.constant(["M"]),
        "Age": tf.constant(["26-35"]),
        "Occupation": tf.constant(["0"]),
        "City_Category": tf.constant(["B"]),
        "Stay_In_Current_City_Years": tf.constant(["4+"]),
        "Marital_Status": tf.constant(["0"]),
    }

def train_and_evaluate(
        train_path: str,
        eval_path: str,
        job_dir: str,
        num_epochs: int,
        embedding_dim: int,
        batch_size: int,
        learning_rate: float,
        user_features: str,
        product_features: str,
        scann_num_neighbors: int = 100, # once scann is stored as a savemodel, this becomes fixed and can no longer be changed
        user_layers: str = '[]',
        product_layers: str = '[]',
        common_layers:str = None,
        user_input_embedding_dim:int = None,
        user_input_embedding_l1:float = 0.0,
        user_input_embedding_l2:float = 0.0,
        product_input_embedding_dim:int = None,
        product_input_embedding_l1:float = 0.0,
        product_input_embedding_l2:float = 0.0,
        common_input_embedding_l1 = None,
        common_input_embedding_l2 = None,
        temperature = 1.0,
        trial=None
):
    if trial is None:
        # adjusting job_dir when running HPTUNING
        on_cloud = 'TF_CONFIG' in os.environ
        trial    = '1'
        if on_cloud:
            trial = str(json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', '1'))
    else:
        trial=str(trial)
        
    if not (job_dir.endswith(f"/{trial}") or job_dir.endswith(f"/{trial}/") ): # sometimes we get double nested folders such as gs://JOBDIR/1/1
        job_dir = os.path.join(job_dir,trial)
    
    
    product_id_col_name: List[str] = json.loads(product_features)
    user_features: List[str] = json.loads(user_features)
    if common_layers is not None:
        user_layers: List[int] = json.loads(common_layers)
        product_layers: List[int] = json.loads(common_layers)
    else:
        user_layers: List[int] = json.loads(user_layers)
        product_layers: List[int] = json.loads(product_layers)
    if user_input_embedding_dim is None:
        user_input_embedding_dim = embedding_dim
    if product_input_embedding_dim is None:
        product_input_embedding_dim = embedding_dim
    if common_input_embedding_l1 is not None:
        user_input_embedding_l1 = common_input_embedding_l1
        product_input_embedding_l1 = common_input_embedding_l1
    if common_input_embedding_l2 is not None:
        user_input_embedding_l2 = common_input_embedding_l2
        product_input_embedding_l2 = common_input_embedding_l2
    
    scann_example = {k: examples_by_feature[k] for k in user_features}

    # ---
    # Load the data
    logging.info("Datamanager init...")
    data_mng = util.DataManager(train_path, eval_path, user_features, product_id_col_name,
                                train_batch_size = batch_size, test_batch_size = batch_size, 
                                shuffle_buffer_size = batch_size, cache_test_set = False)
    logging.info("Datamanager init completed!")

    # ---
    # Create the Model
    logging.info(f"Model building...")
    bf_model = model.create_bf_model(
        data_mng.user_unique_values,
        data_mng.product_unique_values, #data_mng.product_unique_ids,
        embedding_dim,
        learning_rate,
        data_mng.tf_product_unique_records, #data_mng.tf_unique_products,
        batch_size,
        user_layers=user_layers,
        product_layers=product_layers,
        user_input_embedding_dim=user_input_embedding_dim,
        user_input_embedding_l1=user_input_embedding_l1,
        user_input_embedding_l2=user_input_embedding_l2,
        product_input_embedding_dim=product_input_embedding_dim,
        product_input_embedding_l1=product_input_embedding_l1,
        product_input_embedding_l2=product_input_embedding_l2,
        temperature=temperature
    )
    logging.info(f"Model building completed!")
    
    checkpoint = tf.train.latest_checkpoint(job_dir)
    if checkpoint is not None:
        logging.info("loading checkpoint...")
        bf_model.load_weights(checkpoint)
        logging.info("checkpoint loaded")
    

    # ---
    # Train the model
    logging.info(f"Model training...")
    checkpoints_path = os.path.join(job_dir, "model_checkpoints")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path,
                                                     save_weights_only=False,
                                                     verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=job_dir)
    latest_epoch=util.get_latest_epoch(job_dir)
    bf_model.fit(data_mng.cached_train, epochs=latest_epoch+num_epochs, initial_epoch=latest_epoch, 
                 validation_data=data_mng.cached_test, callbacks=[cp_callback, tensorboard_callback])
    logging.info(f"Model training completed!")

    # ---
    # ScaNN builder
    logging.info(f"ScaNN building...")
    tf_unique_products = tf.data.Dataset.from_tensor_slices(data_mng.product_unique_records['Product_ID'])
    tf_unique_products_embeddings = tf.data.Dataset.from_tensor_slices(data_mng.product_unique_records).batch(batch_size).map(bf_model.get_product_tower())


    scann = tfrs.layers.factorized_top_k.ScaNN(bf_model.get_user_tower(), num_reordering_candidates=1000, k=scann_num_neighbors)
    scann.index(tf_unique_products_embeddings, tf_unique_products)
    logging.info(f"ScaNN building completed!")

    # noinspection PyProtectedMember
    logging.info(f"Number of candidates inside ScaNN:{scann._identifiers.shape}")

    # Go in prediction before store or fail
    scann_results = scann(scann_example)  # prediction required to store the model
    # noinspection PyProtectedMember
    assert scann._identifiers.shape[0] == len(data_mng.product_unique_ids), f"Not all candidate are stored into ScaNN"

    # ---
    # Store the models
    for model_name, tf_model in zip(['user', 'candidate', 'Scann'],  # note: "Scann" is mandatory with "S" uppercase
                                    [bf_model.get_user_tower(), bf_model.get_product_tower(), scann]):
        export_path = os.path.join(job_dir, model_name)
        logging.info(f"Storing model `{model_name}` at `{export_path}`...")
        tf_model.save(export_path, options=tf.saved_model.SaveOptions(namespace_whitelist=[model_name]))
        logging.info(f"Model `{model_name}` stored!")

    # Store the params
    params_path = os.path.join(job_dir, "params.json")
    with tf.io.gfile.GFile(params_path, "w") as f:
        params_to_store = {
            "train_path": train_path,
            "eval_path": eval_path,
            "job_dir": job_dir,
            "n_epochs": num_epochs,
            "embedding_dim": embedding_dim,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "product_id_col_name": product_id_col_name,
            "scann_shape": str(scann._identifiers.shape),
            "scann_example": str(scann_example),
            "scann_results": str(scann_results),
            "user_features": user_features,
            "user_layers": user_layers
        }
        json.dump(params_to_store, f, indent=4)


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--train-path',
        type=str,
        default="gs://mlteam-ml-specialization-2021-blackfriday/dataset/parsed/202104130952/train.csv",
        help='gcs path to training dataset')
    parser.add_argument(
        '--eval-path',
        type=str,
        default="gs://mlteam-ml-specialization-2021-blackfriday/dataset/parsed/202104130952/test/evalset.csv",
        help='gcs path to evaluation dataset')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=5,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--embedding-dim',
        default=32,
        type=int,
        help='query and candidate embedding dimension, default=32')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    parser.add_argument(
        '--user-features',
        help='list of user features in json format',
        type=str,
        default='["Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status"]')
    parser.add_argument(
        '--product-features',
        help='list of product features in json format',
        type=str,
        default='["Product_ID"]')
    parser.add_argument(
        '--scann-num-neighbors',
        help='number of results returned by scann saved model',
        type=int,
        default=100),
    parser.add_argument(
        '--user-layers',
        help='list of dimensions of layers in the user model',
        type=str,
        default='[]')
    parser.add_argument(
        '--product-layers',
        help='list of dimensions of layers in the product model',
        type=str,
        default='[]')
    parser.add_argument(
        '--common-layers',
        help='list of dimensions of layers in both models, if used, --user-layers and --product-layers are overwritten by this',
        type=str,
        default=None)
    parser.add_argument(
        '--user-input-embedding-dim',
        help='embedding dimension of every user input feature',
        type=int,
        default=None)
    parser.add_argument(
        '--product-input-embedding-dim',
        help='embedding dimension of every product input feature',
        type=int,
        default=None)
    parser.add_argument(
        '--user-input-embedding-l1',
        help='l1 regularization on the embedding of every user input feature',
        type=float,
        default=0.0)
    parser.add_argument(
        '--product-input-embedding-l1',
        help='l1 regularization on the embedding of every product input feature',
        type=float,
        default=0.0)
    parser.add_argument(
        '--user-input-embedding-l2',
        help='l2 regularization on the embedding of every user input feature',
        type=float,
        default=0.0)
    parser.add_argument(
        '--product-input-embedding-l2',
        help='l2 regularization on the embedding of every product input feature',
        type=float,
        default=0.0)
    parser.add_argument(
        '--common-input-embedding-l1',
        help='l1 regularization on the embedding of every user and product input feature',
        type=float,
        default=None)
    parser.add_argument(
        '--common-input-embedding-l2',
        help='l2 regularization on the embedding of every user and product input feature',
        type=float,
        default=None)
    parser.add_argument(
        '--temperature',
        help='softmax temperature used during training',
        type=float,
        default=1.0)
    parser.add_argument(
        '--trial',
        help='trial number to force on training',
        type=int,
        default=None)
    
    
    
    parsed_args, _ = parser.parse_known_args()
    return parsed_args


if __name__ == '__main__':
    logging.info("[bf_debug] Black friday main task starting...")
    args = get_args()
    logging.getLogger().setLevel(logging.INFO)
    delattr(args, "verbosity")
    train_and_evaluate(**vars(args))
