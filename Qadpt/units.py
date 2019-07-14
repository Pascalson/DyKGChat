import tensorflow as tf
import info_data_utils as data_utils
import collections

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

def encode(cell, encoder_inputs, seq_len=None, dtype=tf.float32):

    with variable_scope.variable_scope("embedding_rnn_seq2seq") as scope:
        scope.set_dtype(dtype)

        return tf.nn.static_rnn(
            cell,
            encoder_inputs,
            sequence_length=seq_len,
            dtype=dtype)




def decode(cell, init_state, embedding, decoder_inputs, decoder_inputs_extra, maxlen, feed_prev=False, loop_function=None, dtype=tf.float32):
    
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
                
                if decoder_inputs_extra is not None:#for Wz
                    output, state = cell(emb_inp + decoder_inputs_extra[i], state)
                else:
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
           
                if decoder_inputs_extra is not None:#for Wz
                    output, state = cell(emb_inp + decoder_inputs_extra[i], state)
                else:
                    output, state = cell(emb_inp, state)
                
                outputs.append(output)
                hiddens.append(state)

                with tf.variable_scope('loop', reuse=True):
                    if output is not None:
                        #loop_return = loop_function(output)
                        loop_return = loop_function(state, output)
                if loop_return is not None:
                    if isinstance(loop_return, list):
                        emb_inp, prev_symbol, a1, kdist, Ndist, Rdebug = loop_return
                        a1s.append(a1)
                        kdists.append(kdist)
                        Ndists.append(Ndist)
                        Rdebugs.append(Rdebug)
                        samples.append(prev_symbol)
                    else:
                        emb_inp = loop_return

                i += 1
                if i >= maxlen:
                    break

            return outputs, samples, hiddens, a1s, kdists, Ndists, Rdebugs
