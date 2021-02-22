import tensorflow as tf
import numpy as np
from constants import DNN_PT_SIZE, EMB_PT_SIZE

class WideNDeepEncoder(object):
    """
    Individual Node Embedding Encoder
    encode each node (with id & attribute) into a vector
    """
    def __init__(self, encoder_name, categorial_features, continuous_features, 
                 FLAGS, dense_dims=(512,),
                 use_input_bn=True, activation='leaky_relu',
                 dropout=0., encode_id_num=0, encode_id_dim=32,
                 is_training=True, ps_num=None):
        """
        :param categorial_features: [n_categories, embedding_dim]
        :param continuous_features: int
        """
        self.is_training = is_training
        self.dense_dims = dense_dims
        self.encoder_name = encoder_name
        self.categorial_features = categorial_features
        self.continuous_features = continuous_features
        self.emb_table = {}
        self.use_input_bn = use_input_bn
        self.FLAGS = FLAGS
        self.dropout = dropout
        self.act = None
        self.encode_id_num = encode_id_num
        self.encode_id_dim = encode_id_dim

        if activation == 'leaky_relu':
            self.act = tf.nn.leaky_relu
        elif activation == 'sigmoid':
            self.act = tf.nn.sigmoid
        elif activation == 'tanh':
            self.act = tf.nn.tanh

        emb_partitioner = None
        self.ps_num = ps_num
        if ps_num is not None:
            emb_partitioner = tf.min_max_variable_partitioner(max_partitions=self.ps_num, min_slice_size=EMB_PT_SIZE)

        with tf.variable_scope('emb_table_' + self.encoder_name, reuse=tf.AUTO_REUSE) as scope:
            if self.categorial_features is not None and len(self.categorial_features) > 0:
                for i, (n_categories, dim) in enumerate(self.categorial_features):
                    if emb_partitioner is not None:
                        self.emb_table[i] = tf.get_variable(
                            "emb_lookup_table_{}".format(i), [n_categories, dim],
                            partitioner=emb_partitioner)
                    else:
                        self.emb_table[i] = tf.get_variable(
                            "emb_lookup_table_{}".format(i), [n_categories, dim])
            if self.encode_id_num > 0 and self.encode_id_dim > 0:
                self.emb_table['id'] = tf.get_variable(
                    "emb_lookup_table_id", [self.encode_id_num, self.encode_id_dim]
                )

    def encode(self, x_categorical=None, x_continuous=None, ids=None):
        """
        x_categorical: [K_1, K_2, ..., n_categoricals]
        x_continuous: [K_1, K_2, ..., n_continuous]
        ids: [K_1, K_2, ...]
        """
        partitioner = None
        if self.ps_num is not None:
            partitioner = tf.min_max_variable_partitioner(
                max_partitions=self.ps_num, min_slice_size=DNN_PT_SIZE)
        with tf.variable_scope('encoding_' + self.encoder_name, reuse=tf.AUTO_REUSE,
                               partitioner=partitioner) as scope:

            x_batch = []
            if self.encode_id_num > 0 and self.encode_id_dim > 0:
                assert ids is not None
                # id_idx = tf.string_to_hash_bucket_fast(ids, self.encode_id_num, name='id_to_hash_idx')
                id_emb = tf.nn.embedding_lookup(self.emb_table['id'], ids, name='embedding_lookup_id')
                x_batch.append(id_emb)

            if self.categorial_features is not None and len(self.categorial_features) > 0:
                assert x_categorical is not None

                to_concats_cat = []
                for i, _ in enumerate(self.categorial_features):
                    emb = tf.nn.embedding_lookup(
                        self.emb_table[i], x_categorical[..., i],
                        name='embedding_lookup_{}_{}'.format(self.encoder_name, i))
                    to_concats_cat.append(emb)

                x_categorical = tf.concat(to_concats_cat, axis=-1, name='cate_concat')
                x_batch.append(x_categorical)
            
            if self.continuous_features is not None and self.continuous_features > 0:
                assert x_continuous is not None
            
                if self.use_input_bn:
                    x_continuous = tf.layers.batch_normalization(x_continuous, training=self.is_training, name='input_bn')
                x_batch.append(x_continuous)

            if len(x_batch) > 1:
                x_batch = tf.concat(x_batch, axis=-1, name='concat_cat_n_cont')
            else:
                x_batch = x_batch[0]
             
            for i, dense_dim in enumerate(self.dense_dims):
                x_batch = tf.layers.dense(x_batch, dense_dim, activation=self.act, name='dense_{}'.format(i))
            if self.dropout:
                x_batch = tf.layers.dropout(x_batch, rate=self.dropout, training=self.is_training, name='dropout')
            return x_batch