import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope

def encode(cell, encoder_inputs, seq_len=None, dtype=tf.float32):
    with variable_scope.variable_scope("embedding_rnn_seq2seq") as scope:
        scope.set_dtype(dtype)
        return tf.nn.static_rnn(
            cell,
            encoder_inputs,
            sequence_length=seq_len,
            dtype=dtype)

def decode(cell, init_state, vocab_size, embedding, decoder_inputs, out_proj, maxlen, more_args, mem_args, feed_prev=False, loop_function=None, copy_transform=None, dtype=tf.float32):
    with variable_scope.variable_scope("embedding_rnn_decoder") as scope:
        outputs = []
        hiddens = []
        state = init_state
        if not feed_prev:
            emb_inputs = (embedding_ops.embedding_lookup(embedding, i)
                          for i in decoder_inputs)
            for i, emb_inp in enumerate(emb_inputs):
                if i >= maxlen:
                    break
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                output, state = cell(emb_inp, state)
                outputs.append(output)
                hiddens.append(state)
            return outputs, hiddens, state
        else:
            a1s = []
            kdists = []
            Ndists = []
            Rdebugs = []
            samples = []
            i = 0
            prev = None
            tmp = None
            emb_inp = embedding_ops.embedding_lookup(embedding, decoder_inputs[0])
            while(True):
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                output, state = cell(emb_inp, state)
                outputs.append(output)
                hiddens.append(state)

                with tf.variable_scope('loop', reuse=True):
                    if output is not None:
                        loop_return = loop_function(output, out_proj, embedding)
                        #loop_return = loop_function(state, output)
                if loop_return is not None:
                    emb_inp, prev_symbol = loop_return
                    samples.append(prev_symbol)
                    #emb_inp, prev_symbol, a1, kdist, Ndist, Rdebug = loop_return
                    #a1s.append(a1)
                    #kdists.append(kdist)
                    #Ndists.append(Ndist)
                    #Rdebugs.append(Rdebug)
                i += 1
                if i >= maxlen:
                    break
            return outputs, samples, hiddens, a1s, kdists, Ndists, Rdebugs


def build_out_proj(size, vocab_size, kdim):
    w = tf.get_variable('proj_w', [size, vocab_size])
    b = tf.get_variable('proj_b', [vocab_size])
    return (w, b)

def build_kg_proj(size, kdim):
    return (None, None)

def build_memnet(size, num_layers, kbembed_size, initializer):
    return None, None

def build_transit_mat(size, kdim, edim, initializer):
    return None, None

def hold_graph(kdim, edim, dtype):
    return None, None

def hold_facts(triples_num, kbembed_size, dtype):
    return None

def hold_kg_indices():
    return None
        

def loop_function(prev, out_proj, embedding):
    prev = nn_ops.xw_plus_b(prev, out_proj[0], out_proj[1])
    prev_symbol = math_ops.argmax(prev, axis=1)
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    return [emb_prev, prev_symbol]

def softmax_loss_function(labels, inputs, out_proj, vocab_size):
    labels = tf.reshape(labels, [-1, 1])
    local_w_t = tf.cast(tf.transpose(out_proj[0]), tf.float32)
    local_b = tf.cast(out_proj[1], tf.float32)
    local_inputs = tf.cast(inputs, tf.float32)
    return tf.cast(tf.nn.sampled_softmax_loss(
        weights = local_w_t,
        biases = local_b,
        inputs = local_inputs,
        labels = labels,
        num_sampled = 512,
        num_classes = vocab_size),
        dtype = tf.float32)

def compute_loss(logits, targets, weights, out_proj, vocab_size):
    with ops.name_scope("sequence_loss", logits + targets + weights):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            crossent = softmax_loss_function(target, logit, out_proj, vocab_size)
            log_perp_list.append(crossent * weight)
        log_perps = math_ops.add_n(log_perp_list)
        total_size = math_ops.add_n(weights)
        total_size += 1e-12
        log_perps /= total_size
        cost = math_ops.reduce_sum(log_perps)
        batch_size = array_ops.shape(targets[0])[0]
        return cost / math_ops.cast(batch_size, cost.dtype)


def copymech(logits, out_proj, vocab_size, kdim, args, mem_args, copy_transform):
    outputs, Ndists, kdists, a1s = [], [], [], []
    total_dists = []
    Rdebugs = []
    return outputs, a1s, kdists, Ndists, logits, Rdebugs

def to_check(logits, outputs, out_proj):
    softmax_outputs = []
    argmax_outputs = []
    for j, outs in enumerate(logits):#length_id
        softmax_outputs.append([])
        argmax_outputs.append([])
        for i in range(len(outs)):#batch_id
            projected_out = nn_ops.xw_plus_b(outs[i], \
                out_proj[0], out_proj[1])
            softmax_outputs[j].append(tf.nn.softmax(projected_out))
            argmax_outputs[j].append(math_ops.argmax(projected_out, axis=1))
    return softmax_outputs, argmax_outputs

def enc_state_transform(enc_state, mem_args):
    return enc_state

def copy_transform():
    return None
