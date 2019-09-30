# coding:utf8
from to_words_by_knowledge import read_in_graph
import numpy as np
import pickle
import queue
import code
import copy
import collections

nodes, edges = read_in_graph()

edges = collections.OrderedDict(sorted(edges.items()))
etypes = list(edges.keys())
total_num = 0
for key, values in edges.items():
    total_num += len(values)
print(total_num)

with open('entity2id.txt','w') as fent, \
    open('relation2id.txt','w') as frel, \
    open('triples.txt','w') as ftrip:

    for idx, n in enumerate(nodes):
        fent.write('{}\t{}\n'.format(n, idx))

    for idx, et in enumerate(etypes):
        frel.write('{}\t{}\n'.format(et, idx))

    for key, values in edges.items():
        for v in values:
            ftrip.write('{}\t{}\t{}\n'.format(v[0], v[1], key))
