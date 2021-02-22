import pickle
import numpy as np
from collections import defaultdict, namedtuple, OrderedDict, Sequence


class DuplicateNodeError(Exception):
    pass


# class FeatureTableWithId(object):
#     def __init__(self, ids, data):
#         self.ids = ids
#         self._id2idx = None
#         self._data = data
    
#     @property
#     def id2idx(self):
#         if self._id2idx is None:
#             self._id2idx = {_id: i for i, _id in enumerate(self.ids)}
#         return self._id2idx
    
#     def __getitem__(self, ids):
#         idx = [self.id2idx[_id] for _id in ids]
#         return self._data[idx]

#     def __len__(self):
#         return len(self._data)


class MultiSourceTable(object):
    def __init__(self, **sources):
        """
        sources: a dict of np.ndarrays
            e.g.: {'cat': catgorical_features, 'cont': continuous_table}
        """
        self._sources = sources

    def __getattr__(self, k):
        try:
            return self._sources[k]
        except KeyError:
            raise AttributeError

    def __getitem__(self, idx):
        return self.get_data(idx)

    def __len__(self):
        return len(next(iter(self._sources.values())))

    def get_data(self, idx, fields=None):
        if fields is None:
            return {key: table[idx] for key, table in self._sources.items()}
        else:
            return {key: self._sources[key][idx] for key in fields}


# class MultiSourceTableWithId(FeatureTableWithId):
#     def __init__(self, ids, **sources):
#         self._sources = MultiSourceTable(**sources)
#         super(MultiSourceTableWithId, self).__init__(ids, self._sources)

#     def get_data(self, ids, fields=None):
#         idx = [self.id2idx[_id] for _id in ids]
#         return self._sources.get_data(idx, fields)

#     def get_data_by_idx(self, idx, fields=None):
#         return self._sources.get_data(idx, fields)


def MultiSourceNodeReferences(node_sources, source_indices):
    return MultiSourceTable(source=node_sources, index=source_indices)


class MultiSourceNodeTable(object):
    def __init__(self, node_types, node_indices, sources):
        """
        node_types: an array of strings denoting the type of the node 
        sources: {source: table} e.g.: {'user': user_table}, {'item': item_table}
        """
        self.node_refs = MultiSourceTable(source=node_types, index=node_indices)
        self.sources = sources

    def __len__(self):
        return len(self.node_refs)

    def __getitem__(self, idx):
        refs = self.node_refs[idx]
        source, index = refs['source'], refs['index']
        uniq_sources = np.unique(source)
        if len(uniq_sources) > 1:
            raise ValueError('Use get_from_multi_sources if query data from multiple sources!')
        source = uniq_sources[0]
        return self.get_from_source(source, index)

    @property
    def n_sources(self):
        return len(self.sources)

    def get_from_source(self, source, index):
        return self.sources[source][index]

    def get_from_multi_sources(self, idx):
        refs = self.node_refs[idx]
        source, index = refs['source'], refs['index']
        source2indices = defaultdict(list)
        for s, i in zip(source, index):
            source2indices[s].append(i)
        return {s: (i, self.get_from_source(s, i)) for s, i in source2indices.items()}


class EdgeTable(object):
    def __init__(self, srcs, dests, edge_data=None, n_nodes=None):
        """
        srcs: np.array of shape [n,], dtype np.int/np.int32
        dsts: np.array of shape [n,], dtype np.int/np.int32
        """
        self.src = srcs
        self.dest = dests
        self.data = edge_data
        self.n_nodes = 0 if n_nodes is None else n_nodes
        self.n_nodes = max(np.max(srcs)+1, np.max(dests)+1, self.n_nodes)
        self._src_index = None
        self._dest_index = None
        self._src_unique = None
        self._dest_unique = None
        self.number_of_nodes = len(np.unique(self.src))
    
    @property
    def src_unique(self):
        if self._src_unique is None:
            self._src_unique = np.unique(self.src)
        return self._src_unique

    @property
    def dest_unique(self):
        if self._dest_unique is None:
            self._dest_unique = np.unique(self.dest)
        return self._dest_unique

    def save(self, path):
        data = {key: getattr(self, key) for key in ['src', 'dest', 'data', 'n_nodes', '_src_index', '_dest_index']}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def save_py2(self, path):
        data = {key: getattr(self, key) for key in ['src', 'dest', 'data', 'n_nodes', '_src_index', '_dest_index']}
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=2)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        ret = EdgeTable(data['src'], data['dest'], data['data'], data['n_nodes'])
        ret._src_index = data['_src_index']
        ret._dest_index = data['_dest_index']
        return ret

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.dest[idx], (None if self.data is None else self.data[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def build_src_index(self):
        self._src_index = [[] for _ in range(self.n_nodes)]
        for i, s in enumerate(self.src):
            self._src_index[s].append(i)

    def build_dest_index(self):
        self._dest_index = [[] for _ in range(self.n_nodes)]
        for i, d in enumerate(self.dest):
            self._dest_index[d].append(i)

    def check_index_build(self, index_type='src'):
        if index_type == 'src' and self._src_index is None:
            raise ValueError('src Index is not built!')
        if index_type == 'dest' and self._dest_index is None:
            raise ValueError('dest Index is not built!')

    def get_edges_by_src(self, src, fields=None):
        try:
            index = self._src_index[src]
        except TypeError:
            print(src)

        if fields is None:
            fields = ('src', 'dest', 'data')
        sources = tuple(getattr(self, f) for f in fields)
        return tuple(s[index] if s is not None else None for s in sources)
    
    def get_edges_by_srcs(self, srcs, fields=None):
        """
        srcs: a int, or a list/1d np.array of src idx
        """
        self.check_index_build('src')
        if isinstance(srcs, Sequence) or isinstance(srcs, np.ndarray):
            return [self.get_edges_by_src(s, fields=fields) for s in srcs]
        return self.get_edges_by_src(srcs, fields=fields)

    def reverse(self):
        # print(self.n_nodes)
        tbl = EdgeTable(self.dest, self.src, edge_data=self.data, n_nodes=self.n_nodes)
        if self._dest_index is None:
            self.build_dest_index()
        tbl._src_index = self._dest_index
        tbl._dest_index = self._src_index
        return tbl

    def nodes(self, with_data=False):
        if with_data:
            for i in range(self.number_of_nodes):
                yield i, self._node_table[i]
        else:
            # for i in range(self.number_of_nodes):
            #     yield i
            for i in self.src_unique:
                yield i

Neighbor = namedtuple('Neighbor', ('idx', 'data'))

Edge = namedtuple('Edge', ('src', 'dest', 'data'))


class BaseGraph(object):

    def get_node(self, key):
        """
        get the node data by key
        """
        raise NotImplementedError

    @property
    def number_of_nodes(self):
        raise NotImplementedError

    @property
    def number_of_edges(self):
        raise NotImplementedError

    def get_adjacent_ids(self, key):
        raise NotImplementedError

    def get_adjacent_nodes(self, key, *args, **kwargs):
        raise NotImplementedError


class BipartiteDirectedGraph(BaseGraph):

    def __init__(self, edge_table, names=None):
        """
        node_table: a multisource node table
        edge_table: the edge table, an instance of the EdgeTable
        """
        # self._ids = ids
        # self._id2idx = {_id: i for i, _id in enumerate(self._ids)}
        self._edge_table = edge_table

    def get_node(self, idx):
        return self._node_table[idx]

    def get_adjacent_ids(self, idx):
        """
        if idx is a id (int), return an adjacent idx list/array
        if idx is a list, return a list of adjacent idx list/array
        """
        return self._edge_table.get_edges_by_srcs(idx, fields=('dest',))[0]

    def get_adjacent_nodes(self, idx, with_data=False):
        dest_idx = self.get_adjacent_ids(idx)
        dest_data = None
        if with_data:
            dest_data = self._node_table[dest_idx]
        return Neighbor(dest_idx, dest_data)

    # def add_node(self, node_id, node_data=None):
    #     assert not self.is_finalized()
    #     if node_id in self:
    #         raise DuplicateNodeError
    #     self._id2idx[node_id] = len(self._ids)
    #     self._ids.append(node_id)
    #     self._node_data.append(node_data)
    #     self._adjacency_lists.append([])

    # def add_edge(self, src, dest, edge_data=None):
    #     assert not self.is_finalized()
    #     if src not in self:
    #         self.add_node(src)
    #     if dest not in self:
    #         self.add_node(dest)
    #     src_idx, dest_idx = self._id2idx[src], self._id2idx[dest]
    #     self._adjacency_lists[src_idx].append(Neighbor(dest_idx, edge_data))
    #     self._n_edges += 1

    def nodes(self, with_data=False):
        if with_data:
            for i in range(self.number_of_nodes):
                yield i, self._node_table[i]
        else:
            for i in range(self.number_of_nodes):
                yield i

    def edges(self):
        for i in range(self.number_of_edges):
            yield self._edge_table[i]

    @property
    def number_of_nodes(self):
        return len(self._node_table)

    def __len__(self):
        return len(self._node_table)

    @property
    def number_of_edges(self):
        return len(self._edge_table)
        # return self._n_edges

    def reverse(self):
        return HomoDirectedGraph(self._node_table, self._edge_table.reverse())
