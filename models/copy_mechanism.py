from models.seq2seq import *

def decode(cell, init_state, vocab_size, embedding, decoder_inputs, out_proj, maxlen, more_args, mem_args, feed_prev=False, loop_function=None, copy_transform=None, dtype=tf.float32):
    with variable_scope.variable_scope("embedding_rnn_decoder") as scope:
        outputs = []
        hiddens = []
        state = init_state
        if not feed_prev:
            emb_inputs = \
                (embedding_ops.embedding_lookup(embedding, i)
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
                        loop_return = \
                            loop_function(i, output, out_proj, vocab_size, embedding, more_args, mem_args, copy_transform)
                if loop_return is not None:
                    emb_inp, prev_symbol, a1, kdist, Ndist, Rdebug = loop_return
                    samples.append(prev_symbol)
                    a1s.append(a1)
                    kdists.append(kdist)
                    Ndists.append(Ndist)
                    Rdebugs.append(Rdebug)
                i += 1
                if i >= maxlen:
                    break
            return outputs, samples, hiddens, a1s, kdists, Ndists, Rdebugs


def build_out_proj(size, vocab_size, kdim):
    w = tf.get_variable('proj_w', [size, vocab_size - kdim])
    b = tf.get_variable('proj_b', [vocab_size - kdim])
    return (w, b)


def copy_generic(logit, out_proj, vocab_size, kdim):
    Ndist = tf.nn.softmax(nn_ops.xw_plus_b(
                logit, out_proj[0], out_proj[1]))
    a1 = tf.reshape(tf.gather(Ndist, tf.constant(4), axis=1), [-1, 1])
    Ndist1 = tf.gather(Ndist, list(range(4)), axis=1)
    Ndist2 = tf.zeros(tf.shape(a1))
    Ndist3 = tf.gather(Ndist, list(range(5, vocab_size - kdim)), axis=1)
    Ndist = tf.concat([Ndist1, Ndist2, Ndist3], 1)
    return a1, Ndist


def combine_copy(a1, kdist, Ndist):
    total_dist = tf.concat([a1*kdist, Ndist],1)
    out_symbol = math_ops.argmax(total_dist, axis=1)
    return total_dist, out_symbol


def loop_function(j, logit, out_proj, vocab_size, embedding, args, mem_args, copy_transform):
    a1, out_symbol, kdist, Ndist, Rdebug, _ = copy_transform(j, logit, out_proj, vocab_size, args, mem_args)
    a1 = tf.reshape(a1, [-1])
    emb_prev = embedding_ops.embedding_lookup(embedding, out_symbol)
    return [emb_prev, out_symbol, a1, kdist, Ndist, Rdebug]

def copymech(logits, out_proj, vocab_size, kdim, args, mem_args, copy_transform):
    outputs, Ndists, kdists, a1s = [], [], [], []
    total_dists = []
    Rdebugs = []
    for j, logit in enumerate(logits):
        a1, out_symbol, kdist, Ndist, Rdebug, total_dist = copy_transform(j, logit, out_proj, vocab_size, args, mem_args)
        outputs.append(out_symbol)
        a1s.append(a1)
        kdists.append(kdist)
        Ndists.append(Ndist)
        total_dists.append(total_dist)
        Rdebugs.append(Rdebug)
    return outputs, a1s, kdists, Ndists, total_dists, Rdebugs


def compute_loss(total_dists, targets, weights, out_proj, vocab_size):
    with ops.name_scope("sequence_loss", targets + weights):
        tgbsize = tf.shape(targets[0])[0]
        log_perp_list = []
        for total_dist, target, weight \
                in zip(total_dists, targets, weights):
            labels = tf.one_hot(target, vocab_size)
            crossent = -tf.reduce_sum(labels * tf.log(total_dist + 1e-12), 1)
            log_perp_list.append(crossent)
        log_perps = math_ops.add_n(log_perp_list)
        total_size = math_ops.add_n(weights) + 1e-12
        log_perps /= total_size
        cost = math_ops.reduce_sum(log_perps)
        batch_size = math_ops.cast(tgbsize, cost.dtype)
        return cost / batch_size

def to_check(logits, outputs, out_proj):
    softmax_outputs = [[] for _ in range(len(outputs))]
    argmax_outputs = outputs
    return softmax_outputs, argmax_outputs
