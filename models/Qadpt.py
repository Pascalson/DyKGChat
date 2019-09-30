from models.copy_mechanism import *

def build_transit_mat(size, kdim, edim, initializer):
    print("build transit_mat")
    Tpred_W = tf.Variable(initializer([size, kdim*edim]))
    Tpred_b = tf.Variable(tf.zeros(shape=[kdim*edim]))
    return Tpred_W, Tpred_b

def hold_graph(kdim, edim, dtype):
    S = tf.placeholder(tf.float32, shape = [None, kdim])
    neA = tf.placeholder(tf.float32, \
        shape = [None, kdim, edim, kdim])
    return S, neA

def copy_transform(j, logit, out_proj, vocab_size, more_args, mem_args):
    print("Qadpt: copy_transform")
    kdist, Rdebug = reasoning(logit, *more_args)
    a1, Ndist = copy_generic(logit, out_proj, vocab_size, more_args[2])
    total_dist, out_symbol = combine_copy(a1, kdist, Ndist)
    return a1, out_symbol, kdist, Ndist, Rdebug, total_dist

def reasoning(logit, Tpred_W, Tpred_b, kdim, edim, neA, S, hops_num, kgpath_len, kg_proj):
    T = tf.matmul(logit, Tpred_W) + Tpred_b
    Tproj = tf.reshape(T, [-1, kdim, 1, edim])
    Tproj = tf.nn.softmax(Tproj, axis=-1)
    Tbsize = tf.shape(T)[0]
    R = tf.squeeze(tf.matmul(Tproj, neA),[2])
    R = R / (tf.reduce_sum(R, 2, keep_dims=True)+1e-12)
    kdists = [tf.reshape(S, [-1, 1, kdim])]
    for _ in range(kgpath_len-1):
        kdists.append(tf.matmul(kdists[-1], R))
    kdist = tf.squeeze(kdists[-1], [1])
    kdist = kdist / (tf.reduce_sum(kdist, 1, keep_dims=True)+1e-12)
    return kdist, [T, Tproj, R, kdists]
