from models.copy_mechanism import *
from models.MemNet import build_memnet, hold_facts, MemNet

def build_kg_proj(size, kdim):
    print("build kg_proj")
    w = tf.get_variable('proj_wk', [size, kdim])
    b = tf.get_variable('proj_bk', [kdim])
    return (w, b)

def copy_transform(j, logit, out_proj, vocab_size, args, mem_args):
    print("TAware: copy_transform")
    _, size, _, hops_num, facts, _, memA, memC = mem_args
    logit, _ = MemNet(logit, size, hops_num, facts, memA, memC)
    kg_proj = args[-1]
    kdist = tf.nn.softmax(nn_ops.xw_plus_b(logit, kg_proj[0], kg_proj[1]))
    a1, Ndist = copy_generic(logit, out_proj, vocab_size, args[2])
    total_dist, out_symbol = combine_copy(a1, kdist, Ndist)
    return a1, out_symbol, kdist, Ndist, kg_proj, total_dist
