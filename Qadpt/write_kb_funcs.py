import math
import numpy as np
import info_data_utils as data_utils

edim = len(data_utils.edge_types)

def print_S(S, fS):
    S_indices = np.nonzero(S)[0]
    for s in S_indices:
        fS.write(data_utils.str_nodes[s])
        fS.write(' ')

def print_neA(neA, fgraph):
    neA_indices = np.nonzero(neA)
    for ne, nt in zip(neA_indices[0], neA_indices[1]):
        ns_idx = int(math.floor(ne/(edim+1)))
        if nt < len(data_utils.str_nodes) and ns_idx < len(data_utils.str_nodes):
            et = int(ne - ns_idx*(edim+1))
            if et == edim:
                e_type = 'ToSelf'
            else:
                e_type = data_utils.edge_types[et]
                fgraph.write(data_utils.str_nodes[ns_idx])
                fgraph.write('-{}-'.format(e_type))
                fgraph.write(data_utils.str_nodes[nt])
                fgraph.write('  |  ')

def print_R(Tproj, neA, fpath):
    R = np.matmul(Tproj, neA)
    R_indices = np.nonzero(R)
    for ns, nt in zip(R_indices[0], R_indices[1]):
        if nt < len(data_utils.str_nodes) and ns < len(data_utils.str_nodes):
            fpath.write('{}-{}'.format(data_utils.str_nodes[ns], data_utils.str_nodes[nt]))
            fpath.write(', ')

def print_debug_R(R, fpath):
    R_indices = np.nonzero(R)
    for ns, nt in zip(R_indices[0], R_indices[1]):
        if nt < len(data_utils.str_nodes) and ns < len(data_utils.str_nodes):
            fpath.write('{}-{}'.format(data_utils.str_nodes[ns], data_utils.str_nodes[nt]))
            fpath.write(', ')

def print_Tproj(Tproj, writer):
    writer.writerow(['entity']+data_utils.edge_types)
    for j, str_n in enumerate(data_utils.str_nodes):
        writer.writerow([str_n]+Tproj[j][0].tolist())
