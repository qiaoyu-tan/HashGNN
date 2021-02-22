import tensorflow as tf
import numpy as np
import BasicModules as bm
from wide_n_deep_encoder import WideNDeepEncoder
from aggregators import MeanAggregator, MaxPoolAggregator, GCNAggregator
from tensorflow.python.framework import ops


def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    More details please refer to https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        #x=tf.clip_by_value(x,-1,1)
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(x)

def _get_shape( T):
    """Return tensor shape, as much as integer."""
    T_shape = T.shape.as_list()
    if not isinstance(T_shape[0], int):
        T_shape[0] = tf.shape(T)[0]
    return T_shape

class BipartiteGraphNodeEncoder(object):
    def __init__(self, i_encoder, u_encoder, encoding_schema='u-i-u', ids=None):
        self.u_encoder = u_encoder
        self.i_encoder = i_encoder
        self.ids = ids
        assert encoding_schema == 'u-i-u' or encoding_schema == 'u-i-i'
        self.encoding_schema = encoding_schema

    def encode_user_layer(self, layer_features):
        """
        :param layer_features: a list of features for each layer. layer-0 is the root node to be finally encoded.
            shape of each element = [-1, i/u_feature_dim]
        :return a list of the same length, all elements are corresponding encoded features
        """
        # [B, L...(max_layer-layer_from_bottom)...L, i_dim]
        ret = []
        for i, layer_feat in enumerate(layer_features):
            if self.encoding_schema == 'u-i-u':
                encoder = self.u_encoder if i % 2 == 0 else self.i_encoder
            else:
                encoder = self.u_encoder if i == 0 else self.i_encoder
            if self.ids is not None:
                encoded = encoder.encode(
                    ids=layer_feat
                )
            else:
                encoded = encoder.encode(
                    x_continuous=layer_feat.get('continuous', None),
                    x_categorical=layer_feat.get('categorical', None),
                    ids=layer_feat.get('ids', None)
                )
            ret.append(encoded)

        return ret

    def encode_item_layer(self, layer_features):
        """
        h_item = aggregate(h_item, h_users')
        :param layer_features: a list of features for each layer. layer-0 is the root node to be finally encoded
        """
        ret = []
        for i, layer_feat in enumerate(layer_features):
            if self.encoding_schema == 'u-i-u':
                encoder = self.i_encoder if i % 2 == 0 else self.u_encoder
            else:
                encoder = self.i_encoder
            if self.ids is not None:
                encoded = encoder.encode(
                    ids=layer_feat
                )
            else:
                encoded = encoder.encode(
                    x_continuous=layer_feat.get('continuous', None),
                    x_categorical=layer_feat.get('categorical', None),
                    ids=layer_feat.get('ids', None)
                )
            ret.append(encoded)
        return ret

def aggregate(layer_features, aggregators):
    """
    h_user = aggregate(h_user, h_items')
    :param layer_features: a list of features for each layer. layer-0 is the root node to be finally encoded.
            shape of each element = [-1, i/u_feature_dim]
    :param aggregators: a list of aggregators, each aggregator for a layer
    """

    assert len(layer_features) == len(aggregators) + 1
    hidden = layer_features

    for layer in range(len(aggregators)):
        next_hidden = []
        aggregator = aggregators[layer]
        for i in range(len(aggregators) - layer):
            self_feature = hidden[i]
            neigh_feature = hidden[i + 1]
            next_hidden.append(aggregator.aggregate(self_feature, neigh_feature))
        hidden = next_hidden
    return hidden[0]


class BiPartiteGraphSAGE(object):
    def __init__(self, FLAGS, global_step, u_categorical_features, u_continuous_features,
                 i_categorical_features, i_continuous_features, graph_input,
                 u_neighs_num=(10, 5), i_neighs_num=(10,), name='gcn',
                 u_id_encode=(0, 0), i_id_encode=(0, 0),
                 mode='train', aggregator='mean', ps_num=None):

        self.u_embed_dim = []
        self.i_embed_dim = []
        self.id_dim = FLAGS.id_dim

        self.u_categorical_features = u_categorical_features
        self.u_continuous_features = u_continuous_features
        self.i_categorical_features = i_categorical_features
        self.i_continuous_features = i_continuous_features
        self.u_id_encode = u_id_encode
        self.i_id_encode = i_id_encode
        if self.u_id_encode[0] > 0:
            self.ids = 1
        else:
            self.ids = None

        self.FLAGS = FLAGS
        self.act = None
        if self.FLAGS.activation == 'leaky_relu':
            self.act = tf.nn.leaky_relu
        elif self.FLAGS.activation == 'sigmoid':
            self.act = tf.nn.sigmoid
        elif self.FLAGS.activation == 'tanh':
            self.act = tf.nn.tanh

        self.is_training = True if mode == 'train' else False
        self.global_step = global_step
        self.i_neighs_num = [] if FLAGS.i_neighs_num == '' else [int(x) for x in FLAGS.i_neighs_num.split(',')]
        self.u_neighs_num = [] if FLAGS.u_neighs_num == '' else [int(x) for x in FLAGS.u_neighs_num.split(',')]
        self.u_embed_dim = [] if FLAGS.u_neighs_num == '' else [int(x) for x in FLAGS.u_embed.split(',')]
        self.i_embed_dim = [] if FLAGS.u_neighs_num == '' else [int(x) for x in FLAGS.i_embed.split(',')]
        self.i_depth = len(self.i_neighs_num)
        self.u_depth = len(self.u_neighs_num)
        self.sparse_k = FLAGS.sparse_k

        if aggregator == 'mean':
            self.Aggregator = MeanAggregator
        elif aggregator == 'max_pool':
            self.Aggregator = MaxPoolAggregator
        elif aggregator == 'gcn':
            self.Aggregator = GCNAggregator
        else:
            assert hasattr(aggregator, 'aggregate')
            self.Aggregator = aggregator

        self.graph_input = graph_input
        self.ps_num = ps_num
        with tf.name_scope(mode):
            self.build_graph(mode=mode)

    def build_graph(self, mode):
        FLAGS = self.FLAGS
        self.u_encoder = WideNDeepEncoder("user",
                                          self.u_categorical_features, self.u_continuous_features,
                                          FLAGS, (self.u_embed_dim[0],),
                                          use_input_bn=FLAGS.use_input_bn,
                                          activation=FLAGS.activation,
                                          dropout=FLAGS.dropout,
                                          encode_id_num=self.u_id_encode[0],
                                          encode_id_dim=self.u_id_encode[1],
                                          is_training=self.is_training, ps_num=self.ps_num)
        self.i_encoder = WideNDeepEncoder("item",
                                          self.i_categorical_features, self.i_continuous_features,
                                          FLAGS, (self.i_embed_dim[0],),
                                          use_input_bn=FLAGS.use_input_bn,
                                          activation=FLAGS.activation,
                                          dropout=FLAGS.dropout,
                                          encode_id_num=self.i_id_encode[0],
                                          encode_id_dim=self.i_id_encode[1],
                                          is_training=self.is_training, ps_num=self.ps_num)

        encoding_schema = FLAGS.encoding_schema
        self.encoder = BipartiteGraphNodeEncoder(
            i_encoder=self.i_encoder,
            u_encoder=self.u_encoder,
            encoding_schema=encoding_schema,
            ids=self.ids)

        u_depth = len(self.u_neighs_num)
        self.u_aggregators = []
        for i in range(u_depth):
            agg = self.Aggregator(input_dim=self.u_embed_dim[i], output_dim=self.u_embed_dim[i + 1],
                                  neigh_input_dim=self.i_embed_dim[i],
                                  name="u_agg_{}".format(i),
                                  act=self.act)
            self.u_aggregators.append(agg)

        i_depth = len(self.i_neighs_num)
        self.i_aggregators = []
        for i in range(i_depth):
            agg = self.Aggregator(input_dim=self.i_embed_dim[i], output_dim=self.i_embed_dim[i + 1],
                                  neigh_input_dim=self.u_embed_dim[i],
                                  name="i_agg_{}".format(i),
                                  act=self.act)
            self.i_aggregators.append(agg)

        # input from queue API, directly parse samples1, samples2, edge_features, type, label
        features = self.graph_input

        self.samples_u = features['samples_u']
        self.samples_i = features['samples_i']
        self.samples_i_neg = features['samples_i_neg']
        self.beta = features['beta']

        if mode == 'export':
            return
        if mode == 'save_emb':
            self.ui_emb()
        else:
            self.loss, self.logits, self.labels = self.ui_model()
            self.pos_prob = tf.sigmoid(self.logits)
            self.pred = tf.concat([tf.reshape(1 - self.pos_prob, [-1, 1]), tf.reshape(self.pos_prob, [-1, 1])],
                                  axis=1, name='concat_prob')

            acc, update_op_acc, precision, update_op_precision, recall, update_op_recall, \
            precision_thr, update_op_precision_thr, recall_thr, update_op_recall_thr, \
            auc_roc, update_op_auc_roc, auc_pr, update_op_auc_pr, confusion_matrix \
                = bm.inner_evaluation(self.pred, self.labels, self.pos_prob, tag="train")

            with tf.name_scope(mode):
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar("accuracy", update_op_acc)
                tf.summary.scalar("precision", update_op_precision)
                tf.summary.scalar("recall", update_op_recall)
                tf.summary.histogram("precision_thr", update_op_precision_thr)
                tf.summary.histogram("recall_thr", update_op_recall_thr)
                tf.summary.scalar("auc_roc", update_op_auc_roc)
                tf.summary.scalar("auc_pr", update_op_auc_pr)
                tf.summary.histogram("confusion_matrix", tf.reshape(confusion_matrix, [1, 4]))

                tf.summary.scalar("accuracy_full", acc)
                tf.summary.scalar("precision_full", precision)
                tf.summary.scalar("recall_full", recall)
                tf.summary.histogram("precision_full_thr", precision_thr)
                tf.summary.histogram("recall_full_thr", recall_thr)
                tf.summary.scalar("auc_roc_full", auc_roc)
                tf.summary.scalar("auc_pr_full", auc_pr)

            self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=mode))

            if mode == 'train':
                learning_rate = FLAGS.learning_rate
                learning_algo = FLAGS.learning_algo
                if learning_algo == "adam":
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                else:
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                if FLAGS.use_input_bn:
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                else:
                    self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                self.train_op = [self.opt_op, self.loss, update_op_auc_roc, self.pos_prob, self.labels,
                                 self.global_step]
            # elif mode == 'eval':
            self.eval_op = [self.loss, update_op_auc_roc, self.pos_prob, self.labels]
            self.batch_eval_op = [self.loss, self.pos_prob, self.labels]

            self.samples = [self.samples_u, self.samples_i, self.labels]

        if mode == 'train':
            self.init_saver()

    def ui_emb(self):

        self.u_model(self.samples_u)

        item_encoded = self.encoder.encode_item_layer(self.samples_i)
        self.item = aggregate(item_encoded, self.i_aggregators)

        neg_item_encoded = self.encoder.encode_item_layer(self.samples_i_neg)
        self.neg_item = aggregate(neg_item_encoded, self.i_aggregators)

        if self.FLAGS.final_dim > 0:
            with tf.variable_scope('final_item_project', reuse=tf.AUTO_REUSE):
                self.item = tf.layers.dense(self.item, self.FLAGS.final_dim, activation=tf.nn.tanh, name='i_proj')
                # self.item = tf.layers.dense(self.item, self.FLAGS.final_dim, activation=tf.nn.sigmoid, name='i_proj')
                self.neg_item = tf.layers.dense(self.neg_item, self.FLAGS.final_dim, activation=tf.nn.tanh, name='i_proj')
                self.user = tf.layers.dense(self.user, self.FLAGS.final_dim, activation=tf.nn.tanh, name='u_proj')
                self.user_hash = binarize(self.user)
                self.item_hash = binarize(self.item)
                self.neg_item_hash = binarize(self.neg_item)

        embs_shape = _get_shape(self.user)
        # embs_left_user = tf.reshape(self.user, [-1, embs_shape[-1]])
        embs_right_hash = tf.reshape(self.user_hash, [-1, embs_shape[-1]])
        selector = self.beta < tf.random_uniform(
            [np.prod(embs_shape[:-1])], dtype=tf.float32)
        self.user_out = tf.where(selector, self.user, embs_right_hash)
        self.user_out = tf.reshape(self.user_out, embs_shape)

        # embs_left_item = tf.reshape(self.user, [-1, embs_shape[-1]])
        embs_right_item_hash = tf.reshape(self.item_hash, [-1, embs_shape[-1]])
        self.item_out = tf.where(selector, self.item, embs_right_item_hash)
        self.item_out = tf.reshape(self.item_out, embs_shape)

        embs_shape_negitem = _get_shape(self.neg_item)
        embs_right_negitem_hash = tf.reshape(self.neg_item_hash, [-1, embs_shape_negitem[-1]])
        selector = self.beta < tf.random_uniform(
            [np.prod(embs_shape_negitem[:-1])], dtype=tf.float32)
        self.neg_item_out = tf.where(selector, self.neg_item, embs_right_negitem_hash)
        self.neg_item_out = tf.reshape(self.neg_item_out, embs_shape_negitem)

    def con_to_spar(self, x_u):
        eye_hot = tf.eye(self.FLAGS.final_dim)
        sparse_k = tf.nn.top_k(x_u, k=self.sparse_k)
        sparse_indices = sparse_k.indices
        sparse_indices = tf.nn.embedding_lookup(eye_hot, sparse_indices)
        sparse_code = tf.reduce_sum(sparse_indices, axis=1)
        return sparse_code

    def con_to_binary(self, x_u):
        ones_ = tf.ones_like(x_u)
        zeros_ = tf.zeros_like(x_u)
        x_u = tf.where(x_u >= 0.5, x=ones_, y=zeros_)
        return x_u

    def u_model(self, user_feats):
        """
        for export
        :return:
        """
        user_encoded = self.encoder.encode_user_layer(user_feats)  # [B, u_dim]
        self.user = aggregate(user_encoded, self.u_aggregators)  # [B, u_dim]

    def ui_model(self):
        self.ui_emb()
        with tf.name_scope('loss'):
            loss, logits, labels = self._xent_loss(self.user_out,
                                                   self.item_out,
                                                   self.neg_item_out)
        return loss, logits, labels


    def rank_dot_product(self, vec1, vec2):
        """
        :param vec1: [B, dim1]
        :param vec2: [B, dim2]
        :return:
        """
        return tf.multiply(vec1, vec2, name='rank_dot_prod')

    def _xent_loss(self, user, item, neg_item):
        """

        :param user: [B, D]
        :param item: [B, D]
        :param neg_item: [B*k, D]
        :return:
        """
        # [B,]
        pos_logit = tf.reduce_sum(self.rank_dot_product(user, item), axis=-1)
        emb_dim = self.FLAGS.final_dim
        if self.FLAGS.final_dim > 0:
            emb_dim = self.FLAGS.final_dim

        # [B * k,]
        neg_logit = tf.reduce_sum(
            tf.multiply(
                tf.reshape(tf.tile(tf.expand_dims(user, axis=1), [1, self.FLAGS.neg_num, 1]), [-1, emb_dim]),
                neg_item
            ),
            axis=-1
        )
        # [B,]
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(pos_logit), logits=pos_logit)
        # [B * k,]
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_logit), logits=neg_logit)
        # loss = tf.reduce_mean(true_xent) + 1.0 * tf.reduce_mean(negative_xent)
        loss = tf.reduce_mean(tf.concat([true_xent, negative_xent], axis=-1))
        logit = tf.concat([pos_logit, neg_logit], axis=-1)
        label = tf.concat([tf.ones_like(pos_logit, dtype=tf.int32), tf.zeros_like(neg_logit, dtype=tf.int32)], axis=-1)
        return loss, logit, label

    def init_saver(self):
        self.saver = tf.train.Saver(sharded=True)


class HashGNN_model(object):
    def __init__(self, FLAGS, global_step, u_categorical_features, u_continuous_features,
                 i_categorical_features, i_continuous_features, graph_input,
                 u_neighs_num=(10, 5), i_neighs_num=(10,), name='gcn',
                 u_id_encode=(0, 0), i_id_encode=(0, 0),
                 mode='train', aggregator='mean', ps_num=None):

        self.u_embed_dim = []
        self.i_embed_dim = []
        self.id_dim = FLAGS.id_dim

        self.u_categorical_features = u_categorical_features
        self.u_continuous_features = u_continuous_features
        self.i_categorical_features = i_categorical_features
        self.i_continuous_features = i_continuous_features
        self.u_id_encode = u_id_encode
        self.i_id_encode = i_id_encode
        if self.u_id_encode[0] > 0:
            self.ids = 1
        else:
            self.ids = None

        self.FLAGS = FLAGS
        self.act = None
        if self.FLAGS.activation == 'leaky_relu':
            self.act = tf.nn.leaky_relu
        elif self.FLAGS.activation == 'sigmoid':
            self.act = tf.nn.sigmoid
        elif self.FLAGS.activation == 'tanh':
            self.act = tf.nn.tanh

        self.is_training = True if mode == 'train' else False
        self.global_step = global_step
        self.i_neighs_num = [] if FLAGS.i_neighs_num == '' else [int(x) for x in FLAGS.i_neighs_num.split(',')]
        self.u_neighs_num = [] if FLAGS.u_neighs_num == '' else [int(x) for x in FLAGS.u_neighs_num.split(',')]
        self.u_embed_dim = [] if FLAGS.u_neighs_num == '' else [int(x) for x in FLAGS.u_embed.split(',')]
        self.i_embed_dim = [] if FLAGS.u_neighs_num == '' else [int(x) for x in FLAGS.i_embed.split(',')]
        self.i_depth = len(self.i_neighs_num)
        self.u_depth = len(self.u_neighs_num)
        self.sparse_k = FLAGS.sparse_k

        if aggregator == 'mean':
            self.Aggregator = MeanAggregator
        elif aggregator == 'max_pool':
            self.Aggregator = MaxPoolAggregator
        elif aggregator == 'gcn':
            self.Aggregator = GCNAggregator
        else:
            assert hasattr(aggregator, 'aggregate')
            self.Aggregator = aggregator

        self.graph_input = graph_input
        self.ps_num = ps_num
        with tf.name_scope(mode):
            self.build_graph(mode=mode)

    def build_graph(self, mode):
        FLAGS = self.FLAGS
        self.u_encoder = WideNDeepEncoder("user",
                                          self.u_categorical_features, self.u_continuous_features,
                                          FLAGS, (self.u_embed_dim[0],),
                                          use_input_bn=FLAGS.use_input_bn,
                                          activation=FLAGS.activation,
                                          dropout=FLAGS.dropout,
                                          encode_id_num=self.u_id_encode[0],
                                          encode_id_dim=self.u_id_encode[1],
                                          is_training=self.is_training, ps_num=self.ps_num)
        self.i_encoder = WideNDeepEncoder("item",
                                          self.i_categorical_features, self.i_continuous_features,
                                          FLAGS, (self.i_embed_dim[0],),
                                          use_input_bn=FLAGS.use_input_bn,
                                          activation=FLAGS.activation,
                                          dropout=FLAGS.dropout,
                                          encode_id_num=self.i_id_encode[0],
                                          encode_id_dim=self.i_id_encode[1],
                                          is_training=self.is_training, ps_num=self.ps_num)

        encoding_schema = FLAGS.encoding_schema
        self.encoder = BipartiteGraphNodeEncoder(
            i_encoder=self.i_encoder,
            u_encoder=self.u_encoder,
            encoding_schema=encoding_schema,
            ids=self.ids)

        u_depth = len(self.u_neighs_num)
        self.u_aggregators = []
        for i in range(u_depth):
            agg = self.Aggregator(input_dim=self.u_embed_dim[i], output_dim=self.u_embed_dim[i + 1],
                                  neigh_input_dim=self.i_embed_dim[i],
                                  name="u_agg_{}".format(i),
                                  act=self.act)
            self.u_aggregators.append(agg)

        i_depth = len(self.i_neighs_num)
        self.i_aggregators = []
        for i in range(i_depth):
            agg = self.Aggregator(input_dim=self.i_embed_dim[i], output_dim=self.i_embed_dim[i + 1],
                                  neigh_input_dim=self.u_embed_dim[i],
                                  name="i_agg_{}".format(i),
                                  act=self.act)
            self.i_aggregators.append(agg)

        # input from queue API, directly parse samples1, samples2, edge_features, type, label
        features = self.graph_input

        self.samples_u = features['samples_u']
        self.samples_i = features['samples_i']
        self.samples_i_neg = features['samples_i_neg']
        self.beta = features['beta']

        if mode == 'export':
            return
        if mode == 'save_emb':
            self.ui_emb()
        else:
            self.loss, self.logits, self.labels = self.ui_model()
            self.pos_prob = tf.sigmoid(self.logits)
            self.pred = tf.concat([tf.reshape(1 - self.pos_prob, [-1, 1]), tf.reshape(self.pos_prob, [-1, 1])],
                                  axis=1, name='concat_prob')

            acc, update_op_acc, precision, update_op_precision, recall, update_op_recall, \
            precision_thr, update_op_precision_thr, recall_thr, update_op_recall_thr, \
            auc_roc, update_op_auc_roc, auc_pr, update_op_auc_pr, confusion_matrix \
                = bm.inner_evaluation(self.pred, self.labels, self.pos_prob, tag="train")

            with tf.name_scope(mode):
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar("accuracy", update_op_acc)
                tf.summary.scalar("precision", update_op_precision)
                tf.summary.scalar("recall", update_op_recall)
                tf.summary.histogram("precision_thr", update_op_precision_thr)
                tf.summary.histogram("recall_thr", update_op_recall_thr)
                tf.summary.scalar("auc_roc", update_op_auc_roc)
                tf.summary.scalar("auc_pr", update_op_auc_pr)
                tf.summary.histogram("confusion_matrix", tf.reshape(confusion_matrix, [1, 4]))

                tf.summary.scalar("accuracy_full", acc)
                tf.summary.scalar("precision_full", precision)
                tf.summary.scalar("recall_full", recall)
                tf.summary.histogram("precision_full_thr", precision_thr)
                tf.summary.histogram("recall_full_thr", recall_thr)
                tf.summary.scalar("auc_roc_full", auc_roc)
                tf.summary.scalar("auc_pr_full", auc_pr)

            self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=mode))

            if mode == 'train':
                learning_rate = FLAGS.learning_rate
                learning_algo = FLAGS.learning_algo
                if learning_algo == "adam":
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                else:
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                if FLAGS.use_input_bn:
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                else:
                    self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                self.train_op = [self.opt_op, self.loss, update_op_auc_roc, self.pos_prob, self.labels,
                                 self.global_step]
            # elif mode == 'eval':
            self.eval_op = [self.loss, update_op_auc_roc, self.pos_prob, self.labels]
            self.batch_eval_op = [self.loss, self.pos_prob, self.labels]

            self.samples = [self.samples_u, self.samples_i, self.labels]

        if mode == 'train':
            self.init_saver()

    def ui_emb(self):

        self.u_model(self.samples_u)

        item_encoded = self.encoder.encode_item_layer(self.samples_i)
        self.item = aggregate(item_encoded, self.i_aggregators)

        neg_item_encoded = self.encoder.encode_item_layer(self.samples_i_neg)
        self.neg_item = aggregate(neg_item_encoded, self.i_aggregators)

        if self.FLAGS.final_dim > 0:
            with tf.variable_scope('final_item_project', reuse=tf.AUTO_REUSE):
                self.item = tf.layers.dense(self.item, self.FLAGS.final_dim, activation=tf.nn.tanh, name='i_proj')
                # self.item = tf.layers.dense(self.item, self.FLAGS.final_dim, activation=tf.nn.sigmoid, name='i_proj')
                self.neg_item = tf.layers.dense(self.neg_item, self.FLAGS.final_dim, activation=tf.nn.tanh, name='i_proj')
                self.user = tf.layers.dense(self.user, self.FLAGS.final_dim, activation=tf.nn.tanh, name='u_proj')
                self.user_hash = binarize(self.user)
                self.item_hash = binarize(self.item)
                self.neg_item_hash = binarize(self.neg_item)

        embs_shape = _get_shape(self.user)
        # embs_left_user = tf.reshape(self.user, [-1, embs_shape[-1]])
        embs_right_hash = tf.reshape(self.user_hash, [-1, embs_shape[-1]])
        selector = self.beta < tf.random_uniform(
            [np.prod(embs_shape[:-1])], dtype=tf.float32)
        self.user_out = tf.where(selector, self.user, embs_right_hash)
        self.user_out = tf.reshape(self.user_out, embs_shape)

        # embs_left_item = tf.reshape(self.user, [-1, embs_shape[-1]])
        embs_right_item_hash = tf.reshape(self.item_hash, [-1, embs_shape[-1]])
        self.item_out = tf.where(selector, self.item, embs_right_item_hash)
        self.item_out = tf.reshape(self.item_out, embs_shape)

        embs_shape_negitem = _get_shape(self.neg_item)
        embs_right_negitem_hash = tf.reshape(self.neg_item_hash, [-1, embs_shape_negitem[-1]])
        selector = self.beta < tf.random_uniform(
            [np.prod(embs_shape_negitem[:-1])], dtype=tf.float32)
        self.neg_item_out = tf.where(selector, self.neg_item, embs_right_negitem_hash)
        self.neg_item_out = tf.reshape(self.neg_item_out, embs_shape_negitem)

    def con_to_spar(self, x_u):
        eye_hot = tf.eye(self.FLAGS.final_dim)
        sparse_k = tf.nn.top_k(x_u, k=self.sparse_k)
        sparse_indices = sparse_k.indices
        sparse_indices = tf.nn.embedding_lookup(eye_hot, sparse_indices)
        sparse_code = tf.reduce_sum(sparse_indices, axis=1)
        return sparse_code

    def con_to_binary(self, x_u):
        ones_ = tf.ones_like(x_u)
        zeros_ = tf.zeros_like(x_u)
        x_u = tf.where(x_u >= 0.5, x=ones_, y=zeros_)
        return x_u

    def u_model(self, user_feats):
        """
        for export
        :return:
        """
        user_encoded = self.encoder.encode_user_layer(user_feats)  # [B, u_dim]
        self.user = aggregate(user_encoded, self.u_aggregators)  # [B, u_dim]

    def ui_model(self):
        self.ui_emb()
        with tf.name_scope('loss'):
            loss, logits, labels = self._xent_loss(self.user_out,
                                                   self.item_out,
                                                   self.neg_item_out)

            mf_loss = self.create_bpr_loss(self.user_out, self.item_out, self.neg_item_out)
            loss = self.FLAGS.lambda_ * loss + mf_loss
        return loss, logits, labels


    def rank_dot_product(self, vec1, vec2):
        """
        :param vec1: [B, dim1]
        :param vec2: [B, dim2]
        :return:
        """
        return tf.multiply(vec1, vec2, name='rank_dot_prod')

    def _xent_loss(self, user, item, neg_item):
        """

        :param user: [B, D]
        :param item: [B, D]
        :param neg_item: [B*k, D]
        :return:
        """
        # [B,]
        pos_logit = tf.reduce_sum(self.rank_dot_product(user, item), axis=-1)
        emb_dim = self.FLAGS.final_dim
        if self.FLAGS.final_dim > 0:
            emb_dim = self.FLAGS.final_dim

        # [B * k,]
        neg_logit = tf.reduce_sum(
            tf.multiply(
                tf.reshape(tf.tile(tf.expand_dims(user, axis=1), [1, self.FLAGS.neg_num, 1]), [-1, emb_dim]),
                neg_item
            ),
            axis=-1
        )
        # [B,]
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(pos_logit), logits=pos_logit)
        # [B * k,]
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_logit), logits=neg_logit)
        loss = tf.reduce_mean(tf.concat([true_xent, negative_xent], axis=-1))
        logit = tf.concat([pos_logit, neg_logit], axis=-1)
        label = tf.concat([tf.ones_like(pos_logit, dtype=tf.int32), tf.zeros_like(neg_logit, dtype=tf.int32)], axis=-1)
        return loss, logit, label

    def init_saver(self):
        self.saver = tf.train.Saver(sharded=True)

    def create_bpr_loss(self, users, pos_items, neg_items):

        emb_dim = self.FLAGS.final_dim
        if self.FLAGS.final_dim > 0:
            emb_dim = self.FLAGS.final_dim

        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        pos_scores = tf.tile(tf.expand_dims(pos_scores, axis=1), [1, self.FLAGS.neg_num])
        pos_scores = tf.reshape(pos_scores, [-1])
        neg_scores = tf.reduce_sum(
            tf.multiply(
                tf.reshape(tf.tile(tf.expand_dims(users, axis=1), [1, self.FLAGS.neg_num, 1]), [-1, emb_dim]),
                neg_items
            ),
            axis=-1
        )
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        return mf_loss

