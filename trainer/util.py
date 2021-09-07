import logging
from typing import List

import pandas as pd
import tensorflow as tf

import os 
from tensorflow.python.summary.summary_iterator import summary_iterator

class DataManager:

    def __init__(self,
                 train_path: str,
                 test_path: str,
                 user_features: List[str],
                 product_id_col_name: List[str],
                 train_batch_size = 2048,
                 test_batch_size = 4096,
                 shuffle_buffer_size = 100_000,
                 cache_test_set = True
                 ):
        tf.random.set_seed(42)

        self.user_features = user_features
        self.product_features = product_id_col_name
        self.all_features = user_features + product_id_col_name
        self.all_features = self.all_features + (["Product_ID"] if "Product_ID" not in self.all_features else [])

        self.train_path = train_path
        self.test_path = test_path
        
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.cache_test_set = cache_test_set

        # Valorized from `_load_data`
        self.cached_train = None
        self.df_train = None
        self.df_test = None
        self._load_data()
        self._parse_data()

        # Create tf dataset
        self.cached_test = None
        self.test_dataset = None
        self.train_dataset = None
        self.make_tf_dataset()

        # Valorized from `_calculate_unique_values`
        self.product_unique_values = None
        self.user_unique_values = None
        self._calculate_unique_values()

        # Job-related work: extract only Product ID info
        self.product_unique_ids = None
        self.tf_unique_products = None
        self._parse_product_id()

    def _parse_product_id(self):
        unique_records=self.df_all[self.product_features+(["Product_ID"] if "Product_ID" not in self.product_features else [])].drop_duplicates()
        # noinspection PyUnresolvedReferences
        #self.product_unique_ids = self.product_unique_values["Product_ID"]
        self.product_unique_ids = unique_records["Product_ID"]
        # noinspection PyUnresolvedReferences
        #self.tf_unique_products = tf.data.Dataset.from_tensor_slices(self.product_unique_values["Product_ID"])
        #self.tf_unique_products = tf.data.Dataset.from_tensor_slices(self.product_unique_values)
        self.product_unique_records = dict(unique_records)
        self.tf_product_unique_records = tf.data.Dataset.from_tensor_slices(dict(self.product_unique_records))

    def _load_data(self):
        # Load the data
        logging.debug(f"Loading training data from {self.train_path}")
        self.df_train = pd.read_csv(tf.io.gfile.GFile(self.train_path, "r"))
        logging.debug(f"Training data loaded, shape: {self.df_train.shape}")

        logging.debug(f"Loading test data from {self.train_path}")
        self.df_test = pd.read_csv(tf.io.gfile.GFile(self.test_path, "r"))
        logging.debug(f"Test data loaded, shape: {self.df_test.shape}")

    def make_tf_dataset(self):
        # Create dataset
        self.test_dataset = tf.data.Dataset.from_tensor_slices(dict(self.df_test))
        self.train_dataset = tf.data.Dataset.from_tensor_slices(dict(self.df_train))  # [!] dict is important

        # Shuffle
        self.train_dataset = self.train_dataset.shuffle(self.shuffle_buffer_size, seed=42, reshuffle_each_iteration=False)

        # Store cached versions
        self.cached_train = self.train_dataset.shuffle(self.shuffle_buffer_size).batch(self.train_batch_size)
        self.cached_test = self.test_dataset.batch(self.test_batch_size)
        if self.cache_test_set:
            self.cached_test = self.cached_test.cache()

    def _parse_data(self):
        """
        Remove unused features from the dataset
        """
        self.df_train = self.df_train[self.all_features]
        logging.debug(f"Parsing data: df_train new shape:`{self.df_train.shape}`")
        self.df_test = self.df_test[self.all_features]
        logging.debug(f"Parsing data: df_test new shape:`{self.df_test.shape}`")

        self.df_train = self.df_train.astype(str)
        self.df_test = self.df_test.astype(str)

    def _calculate_unique_values(self):
        logging.debug(f"Extracting unique values for each feature...")
        df = self.df_train.append(self.df_test)
        self.product_unique_values = {
            feature: df[feature].unique() for feature in self.product_features
        }

        self.user_unique_values = {
            feature: df[feature].unique() for feature in self.user_features
        }
        logging.debug(f"Unique values extraction completed!")
        self.df_all = df

        
        
def get_latest_epoch(job_dir):
    p=os.path.join(job_dir,'train')
    eventsfiles=[os.path.join(p,x) for x in tf.io.gfile.listdir(p) if x.startswith("events")]
    max_step=0
    for eventsfile in eventsfiles:
        for summary in summary_iterator(eventsfile):
            if summary.step > max_step:
                max_step=summary.step
    return max_step