import numpy as np
import math
import pickle
import queue
import code
import copy
import collections

import read_kb

class SSSPs():

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = collections.OrderedDict(sorted(edges.items()))
        self.etypes = list(self.edges.keys())
        self.total_num = 0
        for key, values in self.edges.items():
            self.total_num += len(values)
        self.neighbors_dict = self.get_neighbors()

    def get_neighbors(self):
        neighbors_dict = {}
        for n in self.nodes:
            neighbors_dict[n] = []
        for key, values in self.edges.items():
            for v in values:
                if v[1] not in neighbors_dict[v[0]]:
                    neighbors_dict[v[0]].append(v[1])
        return neighbors_dict

    def relation_in(self, u, v):
        count = 0
        for idx, n in enumerate(v['prev']):
            if n == u['name'] and v['dist'][idx]-1 in u['dist']:
                count += 1
        return count != 0

    def relax(self, u, v, subG, num):
        tmp = sorted([(u['name'], ud + 1) for ud in u['dist']] \
                + [(vp, vd) for vp, vd in zip(v['prev'], v['dist'])], key=lambda x: x[1])
        new_pairs = []
        for pair in tmp:
            if pair not in new_pairs:
                new_pairs.append(pair)
            if len(new_pairs) == num:
                break
        v['prev'] = [p for p, d in new_pairs]
        v['dist'] = [d for p, d in new_pairs]
        if v['name'] in subG:
            subG[v['name']] = list(v['dist'][:len(subG[v['name']])])

    def extract_min(self, subG):
        min_dist = self.total_num
        min_name = 'None'
        for key, dists in subG.items():
            if min(dists) < min_dist:
                min_dist = min(dists)
                min_name = key
        if min_name != 'None':
            if len(subG[min_name]) > 1:
                subG[min_name].pop(subG[min_name].index(min_dist))
            else:
                subG.pop(min_name, None)
        return min_name


    def graph_init(self, s, num):
        G, subG = [], {}
        for idx, n in enumerate(self.nodes):
            if n == s:
                G.append({'name':s, 'dist':[0 for _ in range(num)], 'prev':['source' for _ in range(num)]})
            else:
                G.append({'name':n, 'dist':[self.total_num for _ in range(num)], 'prev':['None' for _ in range(num)]})
            subG[n] = list(G[idx]['dist'])
        return G, subG

    def run(self, s, t, num):#single source shortest paths
        G, subG = self.graph_init(s, num)
        end_list = []
        while len(subG) > 0:
            u = self.extract_min(subG)
            if u == 'None':
                break
            for v in self.neighbors_dict[u]:
                self.relax(G[self.nodes.index(u)], G[self.nodes.index(v)], subG, num)

        paths, dists = [], []
        q = queue.Queue()
        t_id = self.nodes.index(t)
        for v, dist in zip(G[t_id]['prev'], G[t_id]['dist']):
            if v == 'None':
                continue
            if v == 'source':
                if (t,t) not in paths:
                    paths.append((t,t))
                    dists.append(dist)
                break
            if (v,t) not in paths:
                paths.append((v,t))
                dists.append(dist)
            q.put((v, dist))
        while not q.empty():
            v, vd = q.get()
            v_id = self.nodes.index(v)
            for u, dist in zip(G[v_id]['prev'], G[v_id]['dist']):
                if u == 'None':
                    continue
                elif u == 'source':
                    break
                elif dist + 1 == vd:
                    q.put((u, dist))
                    if (u, v) not in paths:
                        paths.append((u,v))

        return paths, dists 


def get_stat(ftxt, finfo, nodes, sp_op, dist_stat):
    lines = ftxt.readlines()
    infos = pickle.load(finfo)
    outs = [lines[2*i+1].strip() for i in range(math.floor(len(lines)/2))]
    for j, (sources, out) in enumerate(zip(infos, outs)):
        sources = [nodes[s] for s in sources]
        targets = [i for i in out.split() if i in nodes]
        for t in targets:
            for s in sources:
                _, dist = sp_op.run(s, t, 1)
                for d in dist:
                    if d not in dist_stat:
                        dist_stat[d] = 1
                    else:
                        dist_stat[d] += 1
                print('complete {}/{}'.format(j, len(outs)))


import json

if __name__ == '__main__':
    nodes, edges = read_kb.read_in_graph('.')
    sp_op = SSSPs(nodes, edges)
    """for test"""
    test_source = '甄嬛'#Zhen-Huan#TODO
    test_target = '皇帝'#the Emperor#TODO
    paths = sp_op.run(test_source, test_target, 5)
    print(paths)

    dist_stat = {}
    with open('train_hgzhz.txt','r') as ftrain, \
        open('train_hgzhz.info','rb') as ftrain_info, \
        open('dev_hgzhz.txt','r') as fvalid, \
        open('dev_hgzhz.info','rb') as fvalid_info, \
        open('test_hgzhz.txt','r') as ftest, \
        open('test_hgzhz.info','rb') as ftest_info, \
        open('stat_dist_sssp.json', 'w') as fout:
        get_stat(ftrain, ftrain_info, nodes, sp_op, dist_stat)
        get_stat(fvalid, fvalid_info, nodes, sp_op, dist_stat)
        get_stat(ftest, ftest_info, nodes, sp_op, dist_stat)
        json.dump(dist_stat, fout, indent=2)
