#encoding=utf8
import jieba
import numpy as np
from numpy.linalg import inv
import os
import copy
import collections
from compute_knowledge import *

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
        
    def set_M(self):
        self.M = get_main_matrix(self.current_graph[0], self.current_graph[1])

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

    def reset(self):
        self.current_graph = self.original_graph
        self.set_M()
    
    def clear_edge(self, name):
        if name not in self.current_graph[1]:
            print("the specified edge type does not exist.")
            return
        self.current_graph[1][name] = []
        self.set_M()

    def delete_edge(self, name, pair):
        if name not in self.current_graph[1]:
            print("the specified edge type does not exist.")
            return
        if pair not in self.current_graph[1][name]:
            print("the specified edge does not exist.")
        self.current_graph[1][name].remove(pair)
        self.set_M()

    def add_edge(self, name, pair):
        if name not in self.current_graph[1]:
            print("the specified edge type does not exist.")
            return
        self.current_graph[1][name].append(pair)
        self.set_M()
