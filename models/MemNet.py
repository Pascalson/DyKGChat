from models.seq2seq import *

def build_memnet(size, num_layers, kbembed_size, initializer):
    print("build MemNet params")
    memA = tf.Variable(initializer([size*num_layers, kbembed_size]))
    memC = tf.Variable(initializer([size*num_layers, kbembed_size]))
    return memA, memC

def hold_facts(triples_num, kbembed_size, dtype):
    facts = tf.placeholder(tf.float32, \
        shape = [None, triples_num, kbembed_size])
    return facts

def MemNet(enc_state, query_size, hops_num, facts, memA, memC):
    print("MemNet: MemNet")
    query = tf.reshape(enc_state, [-1, 1, query_size])
    for _ in range(hops_num):
        M = tf.tensordot(facts, \
            tf.transpose(memA), [[2],[0]])# dim:?xnxd
        M = tf.transpose(M, [0, 2, 1])
        P = tf.matmul(query, M)# dim:?x1xn
        P = tf.nn.softmax(P)
        C = tf.tensordot(facts, \
            tf.transpose(memC), [[2],[0]])# dim:?xnxd
        mem_out = tf.matmul(P, C)
        query = query + mem_out
    return tf.reshape(query, [-1, query_size]), tf.squeeze(P,[1])

def enc_state_transform(enc_state, mem_args):
    print("MemNet: enc_state_transform")
    _, size, num_layers, hops_num, facts, _, memA, memC = mem_args
    mem_out, _ = MemNet(enc_state, size*num_layers, hops_num, facts, memA, memC)
    return tf.tuple([tf.slice(mem_out, [0, l*size], [-1, size]) for l in range(num_layers)])
