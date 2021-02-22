import warnings
from collections import namedtuple, Sequence
import tensorflow as tf
import time
import numpy as np
import random

"""
Classes that are used to sample node neighborhoods
"""

NodeBatch = namedtuple('NodeBatch', ('ids', 'attrs'))

EdgeBatch = namedtuple('EdgeBatch', ('src_ids', 'dst_ids', 'edge_attrs'))

NeighborBatch = namedtuple('NeighborBatch', ('ids', 'attrs'))


class NodeSampler(object):
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size

    def get(self, with_attr=False):
        raise NotImplementedError


class IterateNodeSampler(NodeSampler):
    def __init__(self, graph, batch_size, shuffle=False):
        super(IterateNodeSampler, self).__init__(graph, batch_size)
        self.shuffle = shuffle
        self.it = None

    def get(self, with_attr=False):
        if self.it is None:
            self.it = iter(self.graph.nodes(with_data=with_attr))
        res = []
        for _ in range(self.batch_size):
            try:
                res.append(next(self.it))
            except StopIteration:
                break
        if len(res) == 0:
            raise StopIteration
        if with_attr:
            ids, attrs = zip(*res)
            return NodeBatch(ids, attrs)
        return NodeBatch(np.array(res), None)


class NHopNeighborSampler(object):
    def __init__(self, n_samples_of_hops):
        if isinstance(n_samples_of_hops, int):
            n_samples_of_hops = [n_samples_of_hops]
        assert isinstance(n_samples_of_hops, Sequence)
        self.n_samples_of_hops = n_samples_of_hops

    def get(self, graphs, ids, with_attr=False):
        raise NotImplementedError


def sample(items, n_samples, weights=None):
    try:
        if n_samples < len(items):
            idx = np.random.choice(len(items), n_samples, replace=False, p=weights)
        elif n_samples > len(items):
            idx = np.random.choice(len(items), n_samples, replace=True, p=weights)
        else:
            return items
    except:
        print(items)
        print(n_samples)
        raise
    if isinstance(items, np.ndarray):
        return items[idx]
    else:
        return [items[i] for i in idx]


def map_nested_list(nested_list, f, max_depth=None):
    if max_depth == 0:
        return f(nested_list)
    if isinstance(nested_list, Sequence) or isinstance(nested_list, np.ndarray):
        new_depth = max_depth if max_depth is None else (max_depth - 1)
        return [map_nested_list(e, f, max_depth=new_depth) for e in nested_list]
    return f(nested_list)


class RandomNHopNeighborSampler(NHopNeighborSampler):
    def __init__(self, n_samples_of_hops):
        super(RandomNHopNeighborSampler, self).__init__(n_samples_of_hops)

    def _sample_neighbors(self, edges, ids, n_samples, with_attr=False):
        """
        edges: An Edge Table
        ids: a np.array rerpesenting the ids of the nodes
        n_samples, the number of samples per node

        return: NeighborBatch(ids, attrs)
        """
        ndim = len(ids.shape)

        def _sample_adj(_id):

            _, dest, data = edges.get_edges_by_src(_id)
            try:
                if with_attr:
                    return sample(list(zip(dest, data)), n_samples)
                else:
                    return sample(dest, n_samples)
            except:
                print(_id, dest, data)
                raise

        neighbors = map_nested_list(ids, _sample_adj, ndim)
        if with_attr:
            neighbor_ids = map_nested_list(neighbors, lambda a: a[0], ndim + 1)
            neighbor_attrs = map_nested_list(neighbors, lambda a: a[1], ndim + 1)
            return NeighborBatch(np.array(neighbor_ids), np.array(neighbor_attrs))
        return NeighborBatch(np.array(neighbors), None)

    def get(self, edges_lists, ids, with_attr=False):
        """

        """
        res = []
        for hop, n_samples in enumerate(self.n_samples_of_hops):
            edges = edges_lists[hop] if isinstance(edges_lists, Sequence) else edges_lists
            neighbor_batch = self._sample_neighbors(edges, ids, n_samples, with_attr)
            ids = neighbor_batch.ids
            size_ = np.shape(ids)
            if len(size_) > 2:
                ids = np.reshape(ids, newshape=[-1, size_[-1]])
                neighbor_batch = NeighborBatch(np.array(ids), None)
            res.append(neighbor_batch)
        return res
        # for _id in ids:
        # for i, graph in enumerate(graphs):


class NegativeSampler(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def get(self, graph, ids, with_attr=False):
        raise NotImplementedError


class RandomNegativeSampler(NegativeSampler):
    def __init__(self, n_samples, bipartite=True, replace=True, noisy=True):
        super(RandomNegativeSampler, self).__init__(n_samples)
        self.bipartite = bipartite
        self.replace = replace
        self.noisy = noisy

    def get(self, edges, ids, pool=None, with_attr=False):
        """
        edges: EdgeTable instance, for query the neighbors of given ids
        ids: 1D np.array
        pool: an optional pool of ids for negative sampling

        """
        if not self.bipartite:
            raise NotImplementedError
        original_shape = ids.shape
        ndim = len(ids.shape)
        ids = np.reshape(ids, -1)
        if pool is None:
            pool = edges.dest_unique

        if self.noisy:
            invalid_ids = [edges.get_edges_by_src(_id, fields=('dest',))[0] for _id in ids]
            sample_shape = [len(ids), max([len(v) for v in invalid_ids])]

            sampled = pool[np.random.choice(len(pool), sample_shape, replace=self.replace)]
            assume_unique = not self.replace

            # print('pool', pool.shape, pool.dtype, pool[:10])
            def _sample_negative(i):
                # To speed up set diff calculation
                sampled_ids = np.setdiff1d(sampled[i, :], invalid_ids[i], assume_unique=assume_unique)
                sampled_ids = sample(sampled_ids, self.n_samples)
                if with_attr:
                    data = edges.data[sampled_ids]
                    return sampled_ids, data
                return sampled_ids

            # t1 = time.time()
            neg_samples = [_sample_negative(i) for i in range(len(ids))]
        else:
            neg_samples = pool[np.random.choice(len(pool), (len(ids), self.n_samples), replace=self.replace)]
        # total_t = time.time() - t1
        # print('total {} samples in {:.2f}s, {:.4f}s/sample'.format(len(ids), total_t, total_t/len(ids)))
        if with_attr:
            neg_ids, neg_attr = [np.array(e) for e in zip(*neg_samples)]
            print('neg_ids.shape', neg_ids.shape)
            print('neg_attr.shape', neg_attr.shape)
            return NeighborBatch(neg_ids, neg_attr)

        return NeighborBatch(np.array(neg_samples), None)
