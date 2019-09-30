#encoding=utf8
import numpy as np
import copy

class KGraph:

    def __init__(self, data_dir, read_in_graph):
        self.original_graph = read_in_graph(data_dir)
        self.current_graph = copy.deepcopy(self.original_graph)
        self.set_M()
        self.set_ne()
        self.neA_M = self.get_neA_M()

    def set_ne(self):
        self.ne = []
        et = sorted(self.current_graph[1].keys())
        for key, values in self.current_graph[1].items():
            for value in values:
                self.ne.append((self.current_graph[0].index(value[0]), \
                                et.index(key), \
                                self.current_graph[0].index(value[1])))
            

    def get_neA_M(self):
        kdim = len(self.current_graph[0])
        edim = len(self.current_graph[1])
        neA_M = np.zeros(((kdim+1)*(edim+1), kdim+1))
        for n in range(kdim+1):
            neA_M[n*(edim+1)+edim][-1] = 1.
        for ne in self.ne:
            neA_M[ne[0]*(edim+1)+ne[1]][ne[2]] = 1.
        return neA_M
        
    def make_matrices(self, nodes, edges):
        mats = []
        for key, value in edges.items():
            mats.append(np.zeros((len(nodes), len(nodes))))
            for e in value:
                v1 = nodes.index(e[0])
                v2 = nodes.index(e[1])
                mats[-1][v1][v2] = 1.
                mats[-1][v2][v1] = 1.
        return mats

    def set_M(self):
        nodes, edges = self.current_graph
        mats = self.make_matrices(nodes, edges)
        self.M = np.clip(np.sum(mats, axis=0), 0., 1.)

    def get_vocab_nodes(self):
        nodes = []
        for n in self.original_graph[0]:
            nodes.append(str.encode(n))
        return nodes

    def get_nodes(self):
        return self.current_graph[0]

    def get_edges(self):
        return self.current_graph[1]

    def get_edge_types(self):
        edge_types = sorted(self.current_graph[1])
        return edge_types

    def get_node_edges(self, node):
        edges = {}
        for key, values in self.current_graph[1].items():
            edges[key] = []
            for e in values:
                if node == e[0] or node == e[1]:
                    edges[key].append(e)
        return edges
