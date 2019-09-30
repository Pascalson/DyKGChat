from models.copy_mechanism import *
from models.MemNet import build_memnet, hold_facts, MemNet

def hold_kg_indices():
    return tf.placeholder(tf.int32, shape = [None, 2])

def copy_transform(j, logit, out_proj, vocab_size, args, mem_args):
    batch_size, size, _, hops_num, facts, kg_indices, memA, memC = mem_args
    logit, alpha = MemNet(logit, size, hops_num, facts, memA, memC)
    kdist = tf.Variable(tf.zeros(shape=[batch_size, args[2]+1]), trainable=False, name='kdist{}'.format(j))
    kdist = tf.scatter_nd_add(kdist, kg_indices, tf.reshape(alpha,[-1]))
    kdist = tf.gather(kdist, list(range(args[2])), axis=1)
    kdist = kdist / (tf.reduce_sum(kdist,1, keepdims=True) + 1e-12)
    a1, Ndist = copy_generic(logit, out_proj, vocab_size, args[2])
    total_dist, out_symbol = combine_copy(a1, kdist, Ndist)
    return a1, out_symbol, kdist, Ndist, alpha, total_dist
