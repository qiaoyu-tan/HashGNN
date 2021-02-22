import tensorflow as tf
import numpy as np
import time
import os
import json
import csv
from graph import EdgeTable, MultiSourceTable, MultiSourceNodeTable
from iterators import BatchedIterator
from sampler import IterateNodeSampler, RandomNHopNeighborSampler, RandomNegativeSampler, map_nested_list, \
    NeighborBatch, NodeBatch

DEFAULT_BATCH_SIZE = 32
def interleaving_seq(a, b, length):
    """
    return a,b,a,b,a or a,b,b,b
    :param a:
    :param b:
    :param length:
    :return:
    """
    return [a, b] * (length//2) + [a] * (length % 2)


def get_edge_iterator(edges, batch_size=512, shuffle=False, epochs=-1):
    temp = []
    temp.append(edges.src)
    temp.append(edges.dest)
    edges = np.array(list(zip(*temp)))
    # edges = np.array([(e[0], e[1]) for e in edges])
    batched_iterator = BatchedIterator(edges, batch_size, shuffle=shuffle, epochs=epochs)
    return batched_iterator

def Batch_feat(feat_cat, feat_cont):
    ret = dict()
    ret['categorical'] = feat_cat
    ret['continuous'] = feat_cont

    return ret
def get_category_embedding_size(x):
    if x < 20:
        return 4
    elif x < 1000:
        return 8
    elif x < 10000:
        return 16
    elif x < 100000:
        return 32
    else:
        return 32
        # return 64


class GraphInput(object):
    """
    An Input Manager that iteratively generate input feeds for training
    It takes a bipartite graph as input,
    specifically, ui_edges, ii_edges, u_attrs, and i_attrs
    """

    def __init__(
            self, flag,
            encode_ids=False,
            eval_edges=None
    ):
        self.ui_edges = None
        self.iu_edges = None
        self.ui_edges_eval = None
        self.u_attrs = None
        self.i_attrs = None
        self.u_categoricals = None
        self.i_categoricals = None
        self.u_continuous = None
        self.i_continuous = None
        self.batch_size = flag.batch_size
        self.data_name = flag.dataset
        self.para = flag
        self.neg_num = self.para.neg_num
        self.encode_ids = encode_ids
        self.u_node_iter = None
        self.i_node_iter = None
        self.id_dim = flag.id_dim

        self.i_neighs_num = [] if flag.i_neighs_num == '' else [int(x) for x in flag.i_neighs_num.split(',')]
        self.u_neighs_num = [] if flag.u_neighs_num == '' else [int(x) for x in flag.u_neighs_num.split(',')]
        self.i_depth = len(self.i_neighs_num)
        self.u_depth = len(self.u_neighs_num)

        # self.params = params
        if self.data_name == 'ali':
            self.read_ali_data()
            self.features = self.init_input()
        else:
            # self.read_data_from_csv()
            self.read_data_from_edges()
            self.features = self.init_input_public_dataset()

    def read_ali_data(self):
        file_path_train = os.path.join(os.getcwd(), 'data/' + self.data_name + '/user2item.train_p2.edges')
        file_path_test = os.path.join(os.getcwd(), 'data/' + self.data_name + '/user2item.test_p2.edges')
        user_fea_file_cat = os.path.join(os.getcwd(), 'data/' + self.data_name + '/user_attribute.csv.cat.npy')
        user_fea_file_cont = os.path.join(os.getcwd(), 'data/' + self.data_name + '/user_attribute.csv.cont.npy')
        item_fea_file_cat = os.path.join(os.getcwd(), 'data/' + self.data_name + '/item_attribute.csv.cat.npy')
        item_fea_file_cont = os.path.join(os.getcwd(), 'data/' + self.data_name + '/item_attribute.csv.cont.npy')
        user_meta_file = os.path.join(os.getcwd(), 'data/' + self.data_name + '/user_attribute.csv.meta.json')
        item_meta_file = os.path.join(os.getcwd(), 'data/' + self.data_name + '/item_attribute.csv.meta.json')

        self.ui_edges = EdgeTable.load(file_path_train)
        self.ui_edges_eval = EdgeTable.load(file_path_test)
        self.iu_edges = self.ui_edges.reverse()
        self.len_train_edges = len(self.ui_edges.src)
        user_data_source = {}
        item_data_source = {}

        user_data_source['categorical'] = np.load(user_fea_file_cat, allow_pickle=True)
        user_data_source['continuous'] = np.load(user_fea_file_cont, allow_pickle=True)
        with open(user_meta_file, 'r') as f:
            user_meta = json.load(f)
        mean, std = user_meta['continuous']['mean'], user_meta['continuous']['std']
        # Scale the features to N(0, 1)
        user_data_source['continuous'] = (user_data_source['continuous'] - np.reshape(mean, (1, -1))) / np.reshape(std, (1, -1))

        item_data_source['categorical'] = np.load(item_fea_file_cat, allow_pickle=True)
        item_data_source['continuous'] = np.load(item_fea_file_cont, allow_pickle=True)

        with open(item_meta_file, 'r') as f:
            item_meta = json.load(f)

        mean, std = item_meta['continuous']['mean'], item_meta['continuous']['std']
        # Scale the features to N(0, 1)
        item_data_source['continuous'] = (item_data_source['continuous'] - np.reshape(mean, (1, -1))) / np.reshape(std, (1, -1))

        self.u_categoricals = [
            (len(cat_meta['categories']), get_category_embedding_size(len(cat_meta['categories'])))
            for cat_meta in user_meta['categorical']]
        self.i_categoricals = [
            (len(cat_meta['categories']), get_category_embedding_size(len(cat_meta['categories'])))
            for cat_meta in item_meta['categorical']]

        self.u_continuous = len(user_meta['continuous']['mean'])
        self.i_continuous = len(item_meta['continuous']['mean'])

        temp_dim = 0
        for i, (category, dim) in enumerate(self.u_categoricals):
            temp_dim += dim

        self.dim_user = temp_dim + self.u_continuous
        temp_dim = 0
        for i, (category, dim) in enumerate(self.i_categoricals):
            temp_dim += dim
        self.dim_item = temp_dim + self.i_continuous
        self.is_feature = True

        self.user_len = user_data_source['categorical'].shape[0]
        self.item_len = item_data_source['categorical'].shape[0]

        self.u_attrs = MultiSourceTable(**user_data_source)
        self.i_attrs = MultiSourceTable(**item_data_source)
        self.users = list(self.ui_edges.src_unique)
        self.items = list(self.ui_edges.dest_unique)

        self.print_statistic()

    def read_data_from_edges(self):
        # file_path = os.path.join(os.getcwd(), 'data/' + self.data_name + '_rating.csv')
        file_path_train = os.path.join(os.getcwd(), 'data/' + '{}_user2item.train_p2.edges'.format(self.data_name))
        file_path_test = os.path.join(os.getcwd(), 'data/' + '{}_user2item.test_p2.edges'.format(self.data_name))

        self.ui_edges = EdgeTable.load(file_path_train)
        self.ui_edges_eval = EdgeTable.load(file_path_test)
        user_unique = []
        user_unique.extend(self.ui_edges.src_unique.tolist())
        user_unique.extend(self.ui_edges_eval.src_unique.tolist())
        self.user_len = max(len(set(user_unique)), max(user_unique) + 1)

        user_unique = []
        user_unique.extend(self.ui_edges.dest_unique.tolist())
        user_unique.extend(self.ui_edges_eval.dest_unique.tolist())
        self.item_len = max(len(set(user_unique)), max(user_unique) + 1)

        self.iu_edges = self.ui_edges.reverse()
        self.len_train_edges = len(self.ui_edges.src)
        self.dim_user = self.id_dim
        self.dim_item = self.id_dim
        self.u_categoricals = None
        self.i_categoricals = None
        self.u_continuous = 0
        self.i_continuous = 0
        self.is_feature = False
        print('total train_edges of {}: {} with num_batch={}'.format(self.data_name, self.len_train_edges, int(self.len_train_edges / self.batch_size)))

    def print_statistic(self):

        print('\n loaded dataset={} with user_len={} item_len={} len_train_edges={}'.format(self.data_name,
                                                                                            self.user_len, self.item_len,
                                                                                            self.len_train_edges))

    def init_u_i_iter(self):
        batch_size = self.batch_size
        self.u_node_iter = IterateNodeSampler(self.ui_edges, batch_size=batch_size)
        # self.u_node_iter = BatchedIterator(batch_size=batch_size)
        self.i_node_iter = IterateNodeSampler(self.iu_edges, batch_size=batch_size)

    def init_server(self):
        batch_size = self.batch_size
        self.edge_sampler = get_edge_iterator(self.ui_edges, batch_size=batch_size, epochs=1, shuffle=True)
        self.edge_sampler_eval = get_edge_iterator(self.ui_edges_eval, batch_size=batch_size, epochs=1, shuffle=True)

        self.hop_u_sampler = RandomNHopNeighborSampler(self.u_neighs_num)
        self.hop_i_sampler = RandomNHopNeighborSampler(self.i_neighs_num)
        # self.hop_2_sampler = ss.ByEdgeWeightNeighborSampler(num_at_each_hop=i_neighs_num)
        self.neg_sampler = RandomNegativeSampler(self.neg_num)
        # self.neg_sampler = ss.HardRandomNegativeSampler(neg_num=neg_num)

    def _next_sample(self, edge_sampler, debug=False):
        t1 = time.time()
        edges = edge_sampler.get()

        if debug:
            print('edge sample time {:.2f}s'.format(time.time() - t1))
        src_ids, dest_ids = [np.array(e) for e in zip(*edges)]
        src_attrs = self.u_attrs[src_ids]

        t2 = time.time()

        # src_nbrs = self.hop_u_sampler.get([self.ui_edges], src_ids, with_attr=False)
        src_nbrs = self.hop_u_sampler.get(interleaving_seq(self.ui_edges, self.iu_edges, self.u_depth),
                                          src_ids, with_attr=False)

        if self.u_depth == 1:
            src_nbrs = [NeighborBatch(nbr.ids, self.i_attrs[nbr.ids])
                        for nbr in src_nbrs]
        else:
            temp = []
            for i in range(self.u_depth):
                if i % 2 == 0:
                    temp.extend([NeighborBatch(nbr.ids, self.i_attrs[nbr.ids]) for nbr in [src_nbrs[i]]])
                else:
                    temp.extend([NeighborBatch(nbr.ids, self.u_attrs[nbr.ids]) for nbr in [src_nbrs[i]]])
            src_nbrs = temp

        if debug:
            print('u neighbor sample time {:.2f}s'.format(time.time() - t2))
        dest_attrs = self.i_attrs[dest_ids]
        t3 = time.time()

        dest_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.ui_edges, self.i_depth), dest_ids,
                                           with_attr=False)

        if self.i_depth == 1:
            dest_nbrs = [NeighborBatch(nbr.ids, self.u_attrs[nbr.ids])
                        for nbr in dest_nbrs]
        else:
            temp = []
            for i in range(self.i_depth):
                if i % 2 == 0:
                    temp.extend([NeighborBatch(nbr.ids, self.u_attrs[nbr.ids]) for nbr in [dest_nbrs[i]]])
                else:
                    temp.extend([NeighborBatch(nbr.ids, self.i_attrs[nbr.ids]) for nbr in [dest_nbrs[i]]])
            dest_nbrs = temp


        if debug:
            print('i neighbor sample time {:.2f}s'.format(time.time() - t3))

        t4 = time.time()
        neg_nodes = self.neg_sampler.get(self.ui_edges, src_ids, with_attr=False)
        if debug:
            print('negative sample time {:.2f}s'.format(time.time() - t4))

        neg_ids = np.reshape(neg_nodes.ids, (-1,))

        neg_attrs = self.i_attrs[neg_ids]

        t5 = time.time()
        neg_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.ui_edges, self.i_depth), neg_ids,
                                          with_attr=False)
        if self.i_depth == 1:
            neg_nbrs = [NeighborBatch(nbr.ids, self.u_attrs[nbr.ids])
                        for nbr in neg_nbrs]
        else:
            temp = []
            for i in range(self.i_depth):
                if i % 2 == 0:
                    temp.extend([NeighborBatch(nbr.ids, self.u_attrs[nbr.ids]) for nbr in [neg_nbrs[i]]])
                else:
                    temp.extend([NeighborBatch(nbr.ids, self.i_attrs[nbr.ids]) for nbr in [neg_nbrs[i]]])
            neg_nbrs = temp

        if debug:
            print('i negtive neighbor sample time {:.2f}s'.format(time.time() - t5))
            print('total sample time {:.2f}s'.format(time.time() - t1))

        return NodeBatch(src_ids, src_attrs), src_nbrs, NodeBatch(dest_ids, dest_attrs), dest_nbrs, NodeBatch(neg_ids,
                                                                                                              neg_attrs), neg_nbrs

    def _next_sample_public(self, edge_sampler, debug=False):
        t1 = time.time()
        edges = edge_sampler.get()
        src_ids, dest_ids = [np.array(e) for e in zip(*edges)]

        src_attrs = None

        t2 = time.time()

        src_nbrs = self.hop_u_sampler.get(interleaving_seq(self.ui_edges, self.iu_edges, self.u_depth),
                                          src_ids, with_attr=False)
        if debug:
            print('u neighbor sample time {:.2f}s'.format(time.time() - t2))
        dest_attrs = None
        t3 = time.time()

        dest_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.ui_edges, self.i_depth)
                                           , dest_ids, with_attr=False)

        if debug:
            print('i neighbor sample time {:.2f}s'.format(time.time() - t3))

        t4 = time.time()
        neg_nodes = self.neg_sampler.get(self.ui_edges, src_ids, with_attr=False)
        if debug:
            print('negative sample time {:.2f}s'.format(time.time() - t4))

        neg_ids = np.reshape(neg_nodes.ids, (-1,))

        neg_attrs = None

        t5 = time.time()
        neg_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.ui_edges, self.i_depth),
                                          neg_ids, with_attr=False)
        if debug:
            print('i negtive neighbor sample time {:.2f}s'.format(time.time() - t5))
            print('total sample time {:.2f}s'.format(time.time() - t1))

        return NodeBatch(src_ids, src_attrs), src_nbrs, NodeBatch(dest_ids, dest_attrs),\
               dest_nbrs, NodeBatch(neg_ids, neg_attrs), neg_nbrs

    def _next_user_public(self):
        nodes = self.u_node_iter.get(with_attr=False)
        src_ids = nodes.ids
        src_attrs = None
        dest_ids = np.array(self.ui_edges.dest_unique[0:src_ids.shape[0]])
        dest_attrs = None
        dest_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.ui_edges, self.i_depth)
                                           , dest_ids, with_attr=False)

        src_nbrs = self.hop_u_sampler.get(interleaving_seq(self.ui_edges, self.iu_edges, self.u_depth)
                                          , src_ids, with_attr=False)

        neg_nodes = self.neg_sampler.get(self.ui_edges, src_ids, with_attr=False)

        neg_ids = np.reshape(neg_nodes.ids, (-1,))

        neg_attrs = None

        t5 = time.time()
        neg_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.ui_edges, self.i_depth),
                                          neg_ids, with_attr=False)

        return NodeBatch(src_ids, src_attrs), src_nbrs, NodeBatch(dest_ids, dest_attrs),\
               dest_nbrs, NodeBatch(neg_ids, neg_attrs), neg_nbrs

    def _next_user(self):
        nodes = self.u_node_iter.get(with_attr=False)
        src_ids = nodes.ids
        src_attrs = self.u_attrs[src_ids]
        dest_ids = np.array(range(src_ids.shape[0]))
        dest_attrs = self.i_attrs[dest_ids]
        dest_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.ui_edges, self.i_depth),
                                           dest_ids, with_attr=False)

        if self.i_depth == 1:
            dest_nbrs = [NeighborBatch(nbr.ids, self.u_attrs[nbr.ids])
                        for nbr in dest_nbrs]
        else:
            temp = []
            for i in range(self.i_depth):
                if i % 2 == 0:
                    temp.extend([NeighborBatch(nbr.ids, self.u_attrs[nbr.ids]) for nbr in [dest_nbrs[i]]])
                else:
                    temp.extend([NeighborBatch(nbr.ids, self.i_attrs[nbr.ids]) for nbr in [dest_nbrs[i]]])
            dest_nbrs = temp

        src_nbrs = self.hop_u_sampler.get(interleaving_seq(self.ui_edges, self.iu_edges, self.u_depth)
                                        ,src_ids, with_attr=False)

        if self.u_depth == 1:
            src_nbrs = [NeighborBatch(nbr.ids, self.i_attrs[nbr.ids])
                        for nbr in src_nbrs]
        else:
            temp = []
            for i in range(self.u_depth):
                if i % 2 == 0:
                    temp.extend([NeighborBatch(nbr.ids, self.i_attrs[nbr.ids]) for nbr in [src_nbrs[i]]])
                else:
                    temp.extend([NeighborBatch(nbr.ids, self.u_attrs[nbr.ids]) for nbr in [src_nbrs[i]]])
            src_nbrs = temp

        neg_nodes = self.neg_sampler.get(self.ui_edges, src_ids, with_attr=False)

        neg_ids = np.reshape(neg_nodes.ids, (-1,))

        neg_attrs = self.i_attrs[neg_ids]

        t5 = time.time()
        neg_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.ui_edges, self.i_depth),
                                          neg_ids, with_attr=False)

        if self.i_depth == 1:
            neg_nbrs = [NeighborBatch(nbr.ids, self.u_attrs[nbr.ids])
                        for nbr in neg_nbrs]
        else:
            temp = []
            for i in range(self.i_depth):
                if i % 2 == 0:
                    temp.extend([NeighborBatch(nbr.ids, self.u_attrs[nbr.ids]) for nbr in [neg_nbrs[i]]])
                else:
                    temp.extend([NeighborBatch(nbr.ids, self.i_attrs[nbr.ids]) for nbr in [neg_nbrs[i]]])
            neg_nbrs = temp

        return NodeBatch(src_ids, src_attrs), src_nbrs, NodeBatch(dest_ids, dest_attrs), dest_nbrs,\
               NodeBatch(neg_ids, neg_attrs), neg_nbrs

    def _next_item(self):
        nodes = self.i_node_iter.get(with_attr=False)
        dest_ids = nodes.ids
        src_ids = np.array(range(dest_ids.shape[0]))
        src_attrs = self.u_attrs[src_ids]
        dest_attrs = self.i_attrs[dest_ids]
        dest_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.ui_edges, self.i_depth),
                                           dest_ids, with_attr=False)

        if self.i_depth == 1:
            dest_nbrs = [NeighborBatch(nbr.ids, self.u_attrs[nbr.ids])
                        for nbr in dest_nbrs]
        else:
            temp = []
            for i in range(self.i_depth):
                if i % 2 == 0:
                    temp.extend([NeighborBatch(nbr.ids, self.u_attrs[nbr.ids]) for nbr in [dest_nbrs[i]]])
                else:
                    temp.extend([NeighborBatch(nbr.ids, self.i_attrs[nbr.ids]) for nbr in [dest_nbrs[i]]])
            dest_nbrs = temp

        src_nbrs = self.hop_u_sampler.get(interleaving_seq(self.ui_edges, self.iu_edges, self.u_depth),
                                          src_ids, with_attr=False)

        if self.u_depth == 1:
            src_nbrs = [NeighborBatch(nbr.ids, self.i_attrs[nbr.ids])
                        for nbr in src_nbrs]
        else:
            temp = []
            for i in range(self.u_depth):
                if i % 2 == 0:
                    temp.extend([NeighborBatch(nbr.ids, self.i_attrs[nbr.ids]) for nbr in [src_nbrs[i]]])
                else:
                    temp.extend([NeighborBatch(nbr.ids, self.u_attrs[nbr.ids]) for nbr in [src_nbrs[i]]])
            src_nbrs = temp

        neg_nodes = self.neg_sampler.get(self.ui_edges, src_ids, with_attr=False)

        neg_ids = np.reshape(neg_nodes.ids, (-1,))

        neg_attrs = self.i_attrs[neg_ids]

        t5 = time.time()
        neg_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.ui_edges, self.i_depth),
                                          neg_ids, with_attr=False)

        if self.i_depth == 1:
            neg_nbrs = [NeighborBatch(nbr.ids, self.u_attrs[nbr.ids])
                        for nbr in neg_nbrs]
        else:
            temp = []
            for i in range(self.i_depth):
                if i % 2 == 0:
                    temp.extend([NeighborBatch(nbr.ids, self.u_attrs[nbr.ids]) for nbr in [neg_nbrs[i]]])
                else:
                    temp.extend([NeighborBatch(nbr.ids, self.i_attrs[nbr.ids]) for nbr in [neg_nbrs[i]]])
            neg_nbrs = temp

        return NodeBatch(src_ids, src_attrs), src_nbrs, NodeBatch(dest_ids, dest_attrs), dest_nbrs, \
               NodeBatch(neg_ids, neg_attrs), neg_nbrs

    def _next_item_public(self):
        nodes = self.i_node_iter.get(with_attr=False)
        dest_ids = nodes.ids
        src_ids = np.array(self.ui_edges.src_unique[0:dest_ids.shape[0]])
        src_attrs = None
        dest_attrs = None
        dest_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.ui_edges, self.i_depth),
                                           dest_ids, with_attr=False)
        # dest_nbrs = [NeighborBatch(nbr.ids, None)
        #              for nbr in dest_nbrs]

        src_nbrs = self.hop_u_sampler.get(interleaving_seq(self.ui_edges, self.iu_edges, self.u_depth),
                                          src_ids, with_attr=False)

        # src_nbrs = None

        neg_nodes = self.neg_sampler.get(self.ui_edges, src_ids, with_attr=False)

        neg_ids = np.reshape(neg_nodes.ids, (-1,))

        neg_attrs = None

        t5 = time.time()
        neg_nbrs = self.hop_i_sampler.get(interleaving_seq(self.iu_edges, self.iu_edges, self.i_depth),
                                          neg_ids, with_attr=False)

        # neg_nbrs = [NeighborBatch(nbr.ids, None)
        #             for nbr in neg_nbrs]

        return NodeBatch(src_ids, src_attrs), src_nbrs, NodeBatch(dest_ids, dest_attrs),\
               dest_nbrs, NodeBatch(neg_ids, neg_attrs), neg_nbrs

    def stop(self):
        pass
        # self.server.stop()

    def init_input(self):
        with tf.name_scope('inputs'):
            self.user_hop_features = []
            self.item_hop_features = []
            self.neg_item_hop_features = []
            self.beta_placehoulder = []

            self.beta_placehoulder.append(tf.placeholder(dtype=tf.float32, name='beta'))

            def _create_feature(cat_shape, cont_shape, name_prefix='f'):
                ret = {}
                if cat_shape is not None:
                    ret['categorical'] = tf.placeholder(shape=cat_shape, dtype=tf.int32, name=name_prefix + '_cat')
                if cont_shape is not None:
                    ret['continuous'] = tf.placeholder(shape=cont_shape, dtype=tf.float32, name=name_prefix + '_cont')
                if self.encode_ids:
                    ret['ids'] = tf.placeholder(shape=cat_shape[:-1], dtype=tf.string, name=name_prefix + '_id')
                return ret

            n_u_cats = len(self.u_categoricals) if self.u_categoricals is not None else 0
            n_u_conts = self.u_continuous
            n_i_cats = len(self.i_categoricals) if self.i_categoricals is not None else 0
            n_i_conts = self.i_continuous

            u_neighbor_shapes = []
            for i in range(len(self.u_neighs_num)):
                if i % 2 == 0:
                    u_neighbor_shapes.append((
                        [None, self.u_neighs_num[i], n_i_cats] if n_i_cats > 0 else None,
                        [None, self.u_neighs_num[i], n_i_conts] if n_i_conts > 0 else None))
                else:
                    u_neighbor_shapes.append((
                        [None, self.u_neighs_num[i], n_u_cats] if n_u_cats > 0 else None,
                        [None, self.u_neighs_num[i], n_u_conts] if n_u_conts > 0 else None))

            i_neighbor_shapes = []
            for i in range(len(self.i_neighs_num)):
                if i % 2 == 0:
                    i_neighbor_shapes.append((
                        [None, self.i_neighs_num[i], n_u_cats] if n_u_cats > 0 else None,
                        [None, self.i_neighs_num[i], n_u_conts] if n_u_conts > 0 else None))
                else:
                    i_neighbor_shapes.append((
                        [None, self.i_neighs_num[i], n_i_cats] if n_i_cats > 0 else None,
                        [None, self.i_neighs_num[i], n_i_conts] if n_i_conts > 0 else None))


            self.user_hop_features.append(_create_feature([None, n_u_cats] if n_u_cats > 0 else None,
                [None, n_u_conts] if n_u_conts > 0 else None, 'user_h0'))
            for i in range(len(self.u_neighs_num)):
                self.user_hop_features.append(_create_feature(*u_neighbor_shapes[i],
                                                              name_prefix="user_hop{}".format(i + 1)))

            self.item_hop_features.append(_create_feature(
                [None, n_i_cats] if n_i_cats > 0 else None,
                [None, n_i_conts] if n_i_conts > 0 else None,
                'item_h0'))
            for i in range(len(self.i_neighs_num)):
                self.item_hop_features.append(_create_feature(*i_neighbor_shapes[i],
                                                              name_prefix="item_hop{}".format(i + 1)))

            self.neg_item_hop_features.append(_create_feature(
                [None, n_i_cats] if n_i_cats > 0 else None,
                [None, n_i_conts] if n_i_conts > 0 else None,
                'neg_item_h0'))

            for i in range(len(self.i_neighs_num)):
                self.neg_item_hop_features.append(_create_feature(*i_neighbor_shapes[i],
                                                              name_prefix="neg_item_hop{}".format(i + 1)))
            print('finish placeholder init.')

        ret = {
            'samples_u': self.user_hop_features,  # a list of feature tensors [l0_attrs, l1_nbr_attrs,...,]
            'samples_i': self.item_hop_features,  # a list of feature tensors [l0_attrs, l1_nbr_attrs,...,]
            'samples_i_neg': self.neg_item_hop_features,  # a list of feature tensors [l0_attrs, l1_nbr_attrs,...,]
            'beta': self.beta_placehoulder[0]
        }
        return ret

    def init_input_public_dataset(self):
        with tf.name_scope('inputs'):
            self.user_hop_features = []
            self.item_hop_features = []
            self.neg_item_hop_features = []
            self.beta_placehoulder = []

            u_neighbor_shapes = []
            for i in range(len(self.u_neighs_num)):
                if i % 2 == 0:
                    u_neighbor_shapes.append(
                        [None, self.u_neighs_num[i]])
                else:
                    u_neighbor_shapes.append(
                        [None, self.u_neighs_num[i]])

            i_neighbor_shapes = []
            for i in range(len(self.i_neighs_num)):
                if i % 2 == 0:
                    i_neighbor_shapes.append(
                        [None, self.i_neighs_num[i]])
                else:
                    i_neighbor_shapes.append(
                        [None, self.i_neighs_num[i]])

            self.user_hop_features.append(tf.placeholder(shape=[None], dtype=tf.int32, name='uid'))
            for i in range(self.u_depth):
                self.user_hop_features.append(tf.placeholder(shape=u_neighbor_shapes[i], dtype=tf.int32,
                                                             name='uid_h{}'.format(i+1)))

            self.item_hop_features.append(tf.placeholder(shape=[None], dtype=tf.int32, name='iid'))
            for i in range(self.i_depth):
                self.item_hop_features.append(tf.placeholder(shape=i_neighbor_shapes[i], dtype=tf.int32,
                                                             name='iid_h{}'.format(i+1)))

            self.neg_item_hop_features.append(tf.placeholder(shape=[None], dtype=tf.int32, name='nid'))
            for i in range(self.i_depth):
                self.neg_item_hop_features.append(tf.placeholder(shape=i_neighbor_shapes[i], dtype=tf.int32,
                                                             name='nid_h{}'.format(i+1)))

            self.beta_placehoulder.append(tf.placeholder(dtype=tf.float32, name='beta'))

            print('finish placeholder init.')

        ret = {
            'samples_u': self.user_hop_features,  # a list of feature tensors [l0_attrs, l1_nbr_attrs,...,]
            'samples_i': self.item_hop_features,  # a list of feature tensors [l0_attrs, l1_nbr_attrs,...,]
            'samples_i_neg': self.neg_item_hop_features,  # a list of feature tensors [l0_attrs, l1_nbr_attrs,...,]
            'beta': self.beta_placehoulder[0]
        }
        return ret

    def init_placeholder(self):
        with tf.name_scope('inputs'):
            self.user_hop_features = []
            self.item_hop_features = []
            self.neg_item_hop_features = []

            def _create_feature(cat_shape, cont_shape, name_prefix='f'):
                ret = {}
                if cat_shape is not None:
                    ret['categorical'] = tf.placeholder(shape=cat_shape, dtype=tf.int32, name=name_prefix + '_cat')
                if cont_shape is not None:
                    ret['continuous'] = tf.placeholder(shape=cont_shape, dtype=tf.float32, name=name_prefix + '_cont')
                if self.encode_ids:
                    ret['ids'] = tf.placeholder(shape=cat_shape[:-1], dtype=tf.string, name=name_prefix + '_id')
                return ret
            n_u_cats = len(self.u_categoricals) if self.u_categoricals is not None else 0
            n_u_conts = self.u_continuous
            n_i_cats = len(self.i_categoricals) if self.i_categoricals is not None else 0
            n_i_conts = self.i_continuous

            u_neighbor_shapes = (
                [None, self.u_neighs_num, n_i_cats] if n_i_cats > 0 else None,
                [None, self.u_neighs_num, n_i_conts] if n_i_conts > 0 else None
            )
            i_neighbor_shapes = (
                [None, self.i_neighs_num, n_u_cats] if n_u_cats > 0 else None,
                [None, self.i_neighs_num, n_u_conts] if n_u_conts > 0 else None
            )

            self.user_hop_features = [_create_feature(
                [None, n_u_cats] if n_u_cats > 0 else None,
                [None, n_u_conts] if n_u_conts > 0 else None,
                'user_h0')
                                     ] + [
                                         _create_feature(*u_neighbor_shapes, name_prefix="user_h1"),
                                         # _create_feature(*i_neighbor_shapes, name_prefix="user_h2"),
                                     ]

            self.item_hop_features = [_create_feature(
                [None, n_i_cats] if n_i_cats > 0 else None,
                [None, n_i_conts] if n_i_conts > 0 else None,
                'item_h0')
                                     ] + [
                                         _create_feature(*i_neighbor_shapes, name_prefix="item_h{}".format(1))
                                     ]

            self.neg_item_hop_features = [_create_feature(
                [None, n_i_cats] if n_i_cats > 0 else None,
                [None, n_i_conts] if n_i_conts > 0 else None,
                'neg_item_h0')
                                         ] + [
                                             _create_feature(*i_neighbor_shapes, name_prefix="neg_item_h{}".format(1))
                                         ]

            print('finish placeholder init.')


        ret = {
            'samples_u': self.user_hop_features,  # a list of feature tensors [l0_attrs, l1_nbr_attrs,...,]
            'samples_i': self.item_hop_features,  # a list of feature tensors [l0_attrs, l1_nbr_attrs,...,]
            'samples_i_neg': self.neg_item_hop_features  # a list of feature tensors [l0_attrs, l1_nbr_attrs,...,]
        }
        # print(ret)
        return ret

    def feed_next_sample_train(self):
        if self.data_name == 'ali':
            return self._feed_next_sample(self.edge_sampler)
        else:
            return self._feed_next_sample_public(self.edge_sampler)

    def feed_next_sample_train_continuous(self, beta=1.):
        if self.data_name == 'ali':
            res = self._feed_next_sample(self.edge_sampler)
            res[self.beta_placehoulder[0]] = beta
            return res
        else:
            res = self._feed_next_sample_public(self.edge_sampler)
            res[self.beta_placehoulder[0]] = beta
            return res

    def feed_next_sample_eval(self):
        return self._feed_next_sample(self.edge_sampler_eval)

    def _feed_next_sample_public(self, edge_sampler):
        u_batch, u_nbrs, i_batch, i_nbrs, i_neg_batch, i_neg_nbrs = self._next_sample_public(
            edge_sampler)
        res = dict()
        res[self.user_hop_features[0]] = u_batch.ids
        for i in range(1, self.u_depth + 1):
            res[self.user_hop_features[i]] = u_nbrs[i - 1].ids
        res[self.item_hop_features[0]] = i_batch.ids
        for i in range(1, self.i_depth + 1):
            res[self.item_hop_features[i]] = i_nbrs[i - 1].ids
        res[self.neg_item_hop_features[0]] = i_neg_batch.ids
        for i in range(1, self.i_depth + 1):
            res[self.neg_item_hop_features[i]] = i_neg_nbrs[i - 1].ids
        return res

    def _feed_next_sample(self, edge_sampler):
        u_batch, u_nbrs, i_batch, i_nbrs, i_neg_batch, i_neg_nbrs = self._next_sample(
            edge_sampler)
        # i_neg_attrs = np.reshape(i_neg_attrs, newshape=(-1, 1))
        res = dict()

        def _feed(cat_cont, node_batch):
            # print(cat_cont)
            # print('attrs', type(attrs))
            res[cat_cont['categorical']] = node_batch.attrs['categorical']
            res[cat_cont['continuous']] = node_batch.attrs['continuous']
            if self.encode_ids:
                res[cat_cont['ids']] = np.array(node_batch.ids, dtype=np.bytes_)

        _feed(self.user_hop_features[0], u_batch)
        # print('ushape:',u_attrs.shape)
        for i in range(1, len(self.user_hop_features)):
            _feed(self.user_hop_features[i], u_nbrs[i - 1])


        _feed(self.item_hop_features[0], i_batch)
        for i in range(1, len(self.item_hop_features)):
            _feed(self.item_hop_features[i], i_nbrs[i - 1])

        _feed(self.neg_item_hop_features[0], i_neg_batch)
        for i in range(1, len(self.neg_item_hop_features)):
            _feed(self.neg_item_hop_features[i], i_neg_nbrs[i - 1])
        return res

    def feed_next_user(self):
        if self.data_name == 'ali':
            return self._feed_next_user()
        else:
            return self._feed_next_user_public()

    def feed_next_user_continuous(self, beta=1.):
        if self.data_name == 'ali':
            res, uid, destid = self._feed_next_user()
            res[self.beta_placehoulder[0]] = beta
            return res, uid, destid
        else:
            res, uid, destid = self._feed_next_user_public()
            res[self.beta_placehoulder[0]] = beta
            return res, uid, destid

    def _feed_next_user_public(self):

        u_batch, u_nbrs, i_batch, i_nbrs, i_neg_batch, i_neg_nbrs = self._next_user_public()
        res = dict()
        u_ids = u_batch.ids
        nbr_ids = u_nbrs[0].ids

        res[self.user_hop_features[0]] = u_batch.ids
        for i in range(1, self.u_depth + 1):
            res[self.user_hop_features[i]] = u_nbrs[i - 1].ids
        res[self.item_hop_features[0]] = i_batch.ids
        for i in range(1, self.i_depth + 1):
            res[self.item_hop_features[i]] = i_nbrs[i - 1].ids
        res[self.neg_item_hop_features[0]] = i_neg_batch.ids
        for i in range(1, self.i_depth + 1):
            res[self.neg_item_hop_features[i]] = i_neg_nbrs[i - 1].ids
        return res, u_ids, nbr_ids

    def _feed_next_user(self):
        u_batch, u_nbrs, i_batch, i_nbrs, i_neg_batch, i_neg_nbrs = self._next_user()
        nbr_ids = i_batch.ids
        u_ids = u_batch.ids
        res = dict()

        def _feed(cat_cont, node_batch):
            # print(cat_cont)
            # print('attrs', type(attrs))
            res[cat_cont['categorical']] = node_batch.attrs['categorical']
            res[cat_cont['continuous']] = node_batch.attrs['continuous']
            if self.encode_ids:
                res[cat_cont['ids']] = np.array(node_batch.ids, dtype=np.bytes_)

        _feed(self.user_hop_features[0], u_batch)
        # print('ushape:',u_attrs.shape)
        for i in range(1, len(self.user_hop_features)):
            _feed(self.user_hop_features[i], u_nbrs[i - 1])

        _feed(self.item_hop_features[0], i_batch)
        for i in range(1, len(self.item_hop_features)):
            _feed(self.item_hop_features[i], i_nbrs[i - 1])
            # res[self.item_hop_features[i]] = i_nbrs.hop(i).attrs
            # print(i_nbrs.hop(i).attrs.shape)

        _feed(self.neg_item_hop_features[0], i_neg_batch)
        # res[self.neg_item_hop_features[0]] = i_neg_attrs
        for i in range(1, len(self.neg_item_hop_features)):
            _feed(self.neg_item_hop_features[i], i_neg_nbrs[i - 1])
        return res, u_ids, nbr_ids

    def feed_next_user_test(self):
        nodes = self.u_node_iter.get(with_attr=False)
        src_ids = nodes.ids
        src_attrs = self.u_attrs[src_ids]
        src_nbrs = self.hop_u_sampler.get(interleaving_seq(self.ui_edges, self.iu_edges, self.u_depth)
                                          , src_ids, with_attr=False)

        u_batch = NodeBatch(src_ids, src_attrs)

        if self.u_depth == 1:
            u_nbrs = [NeighborBatch(nbr.ids, self.i_attrs[nbr.ids])
                        for nbr in src_nbrs]
        else:
            temp = []
            for i in range(self.u_depth):
                if i % 2 == 0:
                    temp.extend([NeighborBatch(nbr.ids, self.i_attrs[nbr.ids]) for nbr in [src_nbrs[i]]])
                else:
                    temp.extend([NeighborBatch(nbr.ids, self.u_attrs[nbr.ids]) for nbr in [src_nbrs[i]]])
            u_nbrs = temp

        res = dict()

        def _feed(cat_cont, node_batch):
            res[cat_cont['categorical']] = node_batch.attrs['categorical']
            res[cat_cont['continuous']] = node_batch.attrs['continuous']
            if self.encode_ids:
                res[cat_cont['ids']] = np.array(node_batch.ids, dtype=np.bytes_)

        _feed(self.user_hop_features[0], u_batch)
        for i in range(1, len(self.user_hop_features)):
            _feed(self.user_hop_features[i], u_nbrs[i - 1])
        return res

    def feed_next_item(self):
        if self.data_name == 'ali':
            return self._feed_next_item()
        else:
            return self._feed_next_item_public()

    def feed_next_item_continuous(self, beta=1.):
        if self.data_name == 'ali':
            res, uid, destid = self._feed_next_item()
            res[self.beta_placehoulder[0]] = beta
            return res, uid, destid
        else:
            res, uid, destid = self._feed_next_item_public()
            res[self.beta_placehoulder[0]] = beta
            return res, uid, destid

    def _feed_next_item(self):
        u_batch, u_nbrs, i_batch, i_nbrs, i_neg_batch, i_neg_nbrs = self._next_item()
        nbr_ids = i_batch.ids
        u_ids = u_batch.ids
        res = dict()

        def _feed(cat_cont, node_batch):
            res[cat_cont['categorical']] = node_batch.attrs['categorical']
            res[cat_cont['continuous']] = node_batch.attrs['continuous']
            if self.encode_ids:
                res[cat_cont['ids']] = np.array(node_batch.ids, dtype=np.bytes_)

        _feed(self.user_hop_features[0], u_batch)
        for i in range(1, len(self.user_hop_features)):
            _feed(self.user_hop_features[i], u_nbrs[i - 1])

        _feed(self.item_hop_features[0], i_batch)
        for i in range(1, len(self.item_hop_features)):
            _feed(self.item_hop_features[i], i_nbrs[i - 1])

        _feed(self.neg_item_hop_features[0], i_neg_batch)
        for i in range(1, len(self.neg_item_hop_features)):
            _feed(self.neg_item_hop_features[i], i_neg_nbrs[i - 1])
        return res, u_ids, nbr_ids

    def _feed_next_item_public(self):
        u_batch, u_nbrs, i_batch, i_nbrs, i_neg_batch, i_neg_nbrs = self._next_item_public()
        res = dict()
        u_ids = i_batch.ids
        nbr_ids = i_nbrs[0].ids
        res[self.user_hop_features[0]] = u_batch.ids
        for i in range(1, self.u_depth + 1):
            res[self.user_hop_features[i]] = u_nbrs[i - 1].ids
        res[self.item_hop_features[0]] = i_batch.ids
        for i in range(1, self.i_depth + 1):
            res[self.item_hop_features[i]] = i_nbrs[i - 1].ids
        res[self.neg_item_hop_features[0]] = i_neg_batch.ids
        for i in range(1, self.i_depth + 1):
            res[self.neg_item_hop_features[i]] = i_neg_nbrs[i - 1].ids
        return res, u_ids, nbr_ids

def construct_test_dict(ui_edges_eval):

    train_edges = list(zip(ui_edges_eval.src, ui_edges_eval.dest))
    u2i_test = {i: [] for i in ui_edges_eval.src_unique}
    for i, (src_id, dest_id) in enumerate(train_edges):
        u2i_test[src_id].extend([dest_id])
    return u2i_test

def data_load(data_name):
    file_path_train = os.path.join(os.getcwd(), 'data/' + '{}_user2item.train_p2.edges'.format(data_name))
    file_path_test = os.path.join(os.getcwd(), 'data/' + '{}_user2item.test_p2.edges'.format(data_name))
    ui_edges = EdgeTable.load(file_path_train)
    ui_edges_eval = EdgeTable.load(file_path_test)

    test_dict = construct_test_dict(ui_edges_eval)
    train_dict = construct_test_dict(ui_edges)
    return train_dict, test_dict
