import numpy as np
from numpy.linalg import inv

def make_matrices(nodes, edges):
    mats = []
    for key, value in edges.items():
        mats.append(np.zeros((len(nodes), len(nodes))))
        for e in value:
            v1 = nodes.index(e[0])
            v2 = nodes.index(e[1])
            mats[-1][v1][v2] = 1.
            mats[-1][v2][v1] = 1.
    return mats

def get_main_matrix(nodes, edges):
    mats = make_matrices(nodes, edges)
    M = np.clip(np.sum(mats, axis=0), 0., 1.)
    return M

def get_seq_vec(seq, nodes):
    vec = np.zeros(len(nodes))
    for idx, n in enumerate(nodes):
        if n in seq:
            vec[idx] = 1.
    if np.sum(vec) == 0:
        vec = np.ones(len(nodes))
    return vec

def get_nodes_info(nodes, edges, edge_types):#O(N x E)
    edge_infos = np.zeros((len(nodes), len(edge_types)))
    for nid, n in enumerate(nodes):
        for eid, et in enumerate(edge_types):
            for e in edges[et]:
                if e[0] == n or e[1] == n:
                    edge_infos[nid][eid] = 1.
                    continue
    return edge_infos 

def compute_Laplacian(inp_vec, M):
    D = np.diag(np.sum(M, axis=1))
    L = D - M
    I = np.diag(np.ones(len(M)))
    l = 0.5
    inversed = inv(I+l*L)
    vec = np.dot(inp_vec, inversed)
    return vec

def compute_AKFV(seq, nodes, M):
    inp_vec = get_seq_vec(seq, nodes)
    AKFV = compute_Laplacian(inp_vec, M)
    return AKFV
