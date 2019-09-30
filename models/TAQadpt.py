from models.Qadpt import *
from models.MemNet import build_memnet, hold_facts, MemNet

def copy_transform(j, logit, out_proj, vocab_size, args, mem_args):
    _, size, _, hops_num, facts, _, memA, memC = mem_args
    new_logit, _ = MemNet(logit, size, hops_num, facts, memA, memC)
    kdist, Rdebug = reasoning(logit, *args)
    a1, Ndist = copy_generic(new_logit, out_proj, vocab_size, args[2])
    total_dist, out_symbol = combine_copy(a1, kdist, Ndist)
    return a1, out_symbol, kdist, Ndist, Rdebug, total_dist
