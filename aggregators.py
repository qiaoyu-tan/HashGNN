import numpy as np
import tensorflow as tf
from constants import DNN_PT_SIZE


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    # initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.get_variable(
        initializer=tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32), name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class Aggregator(object):

    def aggregate(self, root_vec, neighbor_vecs):
        raise NotImplementedError


class MeanAggregator(Aggregator):
    """
    Aggregates via mean followed by matmul and non-linearity.
    input_dim : self dim
    output_dim = input_dim
    """

    def __init__(self, input_dim, output_dim,
                 neigh_input_dim=None, is_training=True,
                 bias=False, act=tf.nn.relu,
                 name=None, ps_num=None):

        self.name = name if name is not None else 'mean_agg'
        print('init mean aggregator')
        print('name=', self.name)
        self.bias = bias
        self.act = act
        self.vars = {}

        if neigh_input_dim is None:
            neigh_input_dim = input_dim
        self.neigh_input_dim = neigh_input_dim
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.partitioner = None
        self.ps_num = ps_num
        if self.ps_num is not None:
            self.partitioner = tf.min_max_variable_partitioner(
                max_partitions=self.ps_num, min_slice_size=DNN_PT_SIZE)

        with tf.variable_scope(self.name + '_vars', reuse=tf.AUTO_REUSE,
                               partitioner=self.partitioner):
            self.vars['weights'] = glorot([input_dim + neigh_input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def aggregate(self, self_vecs, neigh_vecs):
        """

        :param self_vecs:  [X, D]
        :param neigh_vecs:  [X * num_nbrs, D]

        :return: a tensor with shape [X, D]
        """
        with tf.name_scope(self.name):
            output_shape = self_vecs.get_shape()
            self_vecs = tf.reshape(self_vecs, [-1, output_shape[-1]])
            neigh_vecs = tf.reshape(neigh_vecs,
                                    [tf.shape(self_vecs)[0], -1, neigh_vecs.shape[-1]])
            neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

            # [nodes] x [out_dim]
            cat = tf.concat([self_vecs, neigh_means], axis=-1, name='cat')

            # bias
            if self.bias:
                output = tf.nn.xw_plus_b(cat, self.vars['weights'], self.vars['bias'])
            else:
                output = tf.matmul(cat, self.vars['weights'])

            if self.act:
                output = self.act(output)
            return output


class GCNAggregator(Aggregator):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 bias=False, act=tf.nn.relu, name=None, ps_num=None):

        self.name = name if name is not None else 'gcn_agg'
        self.bias = bias
        self.act = act

        if neigh_input_dim is None:
            neigh_input_dim = input_dim
        self.neigh_input_dim = neigh_input_dim
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.partitioner = None
        self.ps_num = ps_num
        if self.ps_num is not None:
            self.partitioner = tf.min_max_variable_partitioner(
                max_partitions=self.ps_num, min_slice_size=DNN_PT_SIZE)

        self.vars = {}
        with tf.variable_scope(self.name + '_vars', reuse=tf.AUTO_REUSE,
                               partitioner=self.partitioner):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def aggregate(self, self_vecs, neigh_vecs):
        with tf.name_scope(self.name):
            neigh_vecs = tf.reshape(neigh_vecs,
                                    [tf.shape(self_vecs)[0], -1, neigh_vecs.shape[-1]])

            means = tf.reduce_mean(tf.concat([neigh_vecs,
                                              tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)

            # bias
            if self.bias:
                output = tf.nn.xw_plus_b(means, self.vars['weights'], self.vars['bias'])
            else:
                output = tf.matmul(means, self.vars['weights'])

            if self.act:
                output = self.act(output)
            return output


class MaxPoolAggregator(Aggregator):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 bias=False, act=tf.nn.relu, name=None, ps_num=None):

        self.name = name if name is not None else 'gcn_agg'
        self.bias = bias
        self.act = act

        if neigh_input_dim is None:
            neigh_input_dim = input_dim
        self.neigh_input_dim = neigh_input_dim
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.partitioner = None
        self.ps_num = ps_num
        if self.ps_num is not None:
            self.partitioner = tf.min_max_variable_partitioner(
                max_partitions=self.ps_num, min_slice_size=DNN_PT_SIZE)

        self.vars = {}
        with tf.variable_scope(self.name + '_vars', reuse=tf.AUTO_REUSE,
                               partitioner=self.partitioner):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def aggregate(self, self_vecs, neigh_vecs):
        with tf.name_scope(self.name):
            neigh_vecs = tf.reshape(neigh_vecs,
                                    [tf.shape(neigh_vecs)[0], -1, neigh_vecs.shape[-1]])

            neigh_vecs = tf.reduce_max(neigh_vecs, axis=1)

            # [nodes] x [out_dim]
            cat = tf.concat([self_vecs, neigh_vecs], axis=-1, name='cat')

            # bias
            if self.bias:
                output = tf.nn.xw_plus_b(cat, self.vars['weights'], self.vars['bias'])
            else:
                output = tf.matmul(cat, self.vars['weights'])

            if self.act:
                output = self.act(output)
            return output

# class MeanPoolingAggregator(Layer):
#     """ Aggregates via mean-pooling over MLP functions.
#     """
#     def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
#             dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
#         super(MeanPoolingAggregator, self).__init__(**kwargs)

#         self.dropout = dropout
#         self.bias = bias
#         self.act = act
#         self.concat = concat

#         if neigh_input_dim is None:
#             neigh_input_dim = input_dim

#         if name is not None:
#             name = '/' + name
#         else:
#             name = ''

#         if model_size == "small":
#             hidden_dim = self.hidden_dim = 512
#         elif model_size == "big":
#             hidden_dim = self.hidden_dim = 1024

#         self.mlp_layers = []
#         self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
#                                  output_dim=hidden_dim,
#                                  act=tf.nn.relu,
#                                  dropout=dropout,
#                                  sparse_inputs=False,
#                                  logging=self.logging))

#         with tf.variable_scope(self.name + name + '_vars'):
#             self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
#                                                         name='neigh_weights')

#             self.vars['self_weights'] = glorot([input_dim, output_dim],
#                                                         name='self_weights')
#             if self.bias:
#                 self.vars['bias'] = zeros([self.output_dim], name='bias')

#         if self.logging:
#             self._log_vars()

#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.neigh_input_dim = neigh_input_dim

#     def _call(self, inputs):
#         self_vecs, neigh_vecs = inputs
#         neigh_h = neigh_vecs

#         dims = tf.shape(neigh_h)
#         batch_size = dims[0]
#         num_neighbors = dims[1]
#         # [nodes * sampled neighbors] x [hidden_dim]
#         h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

#         for l in self.mlp_layers:
#             h_reshaped = l(h_reshaped)
#         neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
#         neigh_h = tf.reduce_mean(neigh_h, axis=1)

#         from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
#         from_self = tf.matmul(self_vecs, self.vars["self_weights"])

#         if not self.concat:
#             output = tf.add_n([from_self, from_neighs])
#         else:
#             output = tf.concat([from_self, from_neighs], axis=1)

#         # bias
#         if self.bias:
#             output += self.vars['bias']

#         return self.act(output)