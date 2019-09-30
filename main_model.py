import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import numpy as np
import copy
import importlib

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class main_model:
    def __init__(
            self,
            method,
            model,
            kdim,
            edim,
            kbembed_size,
            triples_num,
            size,
            num_layers,
            vocab_size,
            buckets,
            hops_num=1,#TODO
            kgpath_len=1,#TODO
            learning_rate=0.5,
            learning_rate_decay_factor=0.99,
            max_gradient_norm=5.0,
            feed_prev=False,
            batch_size=32,
            dtype=tf.float32):

        model_funcs = importlib.import_module('models.' + model) 
        globals().update(model_funcs.__dict__)

        # for knowledge graph
        self.kdim = kdim
        self.edim = edim
        self.kbembed_size = kbembed_size
        self.triples_num = triples_num
        self.hops_num = hops_num#TODO
        self.kgpath_len = kgpath_len#TODO

        # basic
        self.size = size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        print('VOCABSIZE:{}'.format(vocab_size))
        self.buckets = buckets
        self.feed_prev = feed_prev
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.op_lr_decay = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        

        # main model
        self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = core_rnn_cell.EmbeddingWrapper(
            cell=self.enc_cell,
            embedding_classes=vocab_size,
            embedding_size=size)
        # input embedding
        self.embedding = variable_scope.get_variable('embedding', [vocab_size, size])

        # encoder's placeholder
        self.encoder_inputs = []
        for bid in range(buckets[-1][0]):
            self.encoder_inputs.append(
                tf.placeholder(tf.int32, shape = [None],
                               name = 'encoder{0}'.format(bid)))
        self.seq_len = tf.placeholder(
            tf.int32, shape = [None],
            name = 'enc_seq_len')
        # decoder's placeholder
        self.decoder_inputs = []
        self.targets = []
        self.target_weights = []
        self.masks = []
        for bid in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(
                tf.placeholder(tf.int32, shape = [None],
                    name = 'decoder{0}'.format(bid)))
            self.targets.append(
                tf.placeholder(tf.int32, shape = [None],
                    name = 'target{0}'.format(bid)))
            self.target_weights.append(
                tf.placeholder(tf.float32, shape = [None],
                    name = 'weight{0}'.format(bid)))
            self.masks.append(
                tf.placeholder(tf.float32, shape = [None],
                    name = 'mask_unit{0}'.format(bid)))

        # TODO passed args funcs
        self.output_projection = build_out_proj(size, vocab_size, kdim)
        self.kg_projection = build_kg_proj(size, kdim)
        self.memA, self.memC = build_memnet(size, num_layers, kbembed_size, xavier_init)
        self.Tpred_W, self.Tpred_b = build_transit_mat(size, kdim, edim, xavier_init)
        self.S, self.neA = hold_graph(kdim, edim, dtype)
        self.facts = hold_facts(triples_num, kbembed_size, dtype)
        self.kg_indices = hold_kg_indices()
        more_args = (self.Tpred_W, self.Tpred_b, self.kdim, self.edim, self.neA, self.S, self.hops_num, self.kgpath_len, self.kg_projection)
        mem_args = (self.batch_size, self.size, self.num_layers, self.hops_num, self.facts, self.kg_indices, self.memA, self.memC)

        if method == 'TRAIN':

            self.enc_state = []
            self.losses = []
            self.logits = []

            self.decKB_losses = []
            self.decN_losses = []
            self.ptr_losses = []
            self.outputs = []
            self.a1s = []
            self.kdists = []
            self.Ndists = []
            self.Rdebugs = []

            for j, bucket in enumerate(buckets):

                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True if j > 0 else None):

                    _, enc_state = \
                        encode(self.enc_cell, self.encoder_inputs[:bucket[0]], self.seq_len)
                    enc_state = enc_state_transform(enc_state, mem_args)

                    logits, hiddens, dec_state = \
                        decode(self.cell, enc_state, \
                               self.vocab_size, self.embedding, \
                               self.decoder_inputs[:bucket[1]], \
                               self.output_projection, \
                               bucket[1]+1, more_args, \
                               None, feed_prev=False, \
                               copy_transform=copy_transform)
                    
                    outputs, a1s, kdists, Ndists, final_logits, Rdebug = copymech(logits, self.output_projection, self.vocab_size, self.kdim, more_args, mem_args, copy_transform)
                    loss = compute_loss(final_logits, self.targets[:bucket[1]], self.target_weights[:bucket[1]], self.output_projection, self.vocab_size)

                    self.enc_state.append(enc_state)
                    self.losses.append(loss)
                    self.logits.append(logits)

                    self.outputs.append(outputs)
                    self.a1s.append(a1s)
                    self.kdists.append(kdists)
                    self.Ndists.append(Ndists)
                    self.Rdebugs.append(Rdebug)

            # TODO check
            self.softmax_outputs, self.argmax_outputs = to_check(self.logits, self.outputs, self.output_projection)

            # update methods
            self.op_update = []
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            params = tf.trainable_variables()
            print(params)
            for j in range(len(self.buckets)):
                gradients = tf.gradients(self.losses[j], params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.op_update.append(optimizer.apply_gradients(zip(clipped_gradients, params),
                                                                global_step=self.global_step))


        elif method == 'TEST':
            self.enc_state = []
            self.argmax_outputs = []
            self.logits = []

            self.a1s = []
            self.kdists = []
            self.Ndists = []
            self.Rdebugs = []

            for j, bucket in enumerate(buckets):

                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True if j > 0 else None):

                    _, enc_state = \
                        encode(self.enc_cell, self.encoder_inputs[:bucket[0]], self.seq_len)
                    enc_state = enc_state_transform(enc_state, mem_args)

                    logits, argmax_outputs, hiddens, a1s, kdists, Ndists, Rdebugs = \
                        decode(self.cell, enc_state, \
                               self.vocab_size, self.embedding, \
                               self.decoder_inputs[:bucket[1]], \
                               self.output_projection, \
                               bucket[1], more_args, \
                               mem_args, feed_prev=True, \
                               loop_function=loop_function, \
                               copy_transform=copy_transform)

                self.enc_state.append(enc_state)
                self.argmax_outputs.append(argmax_outputs)
                self.logits.append(logits)

                self.a1s.append(a1s)
                self.kdists.append(kdists)
                self.Ndists.append(Ndists)
                self.Rdebugs.append(Rdebugs)

            params = tf.trainable_variables()
            print(params)

        # saver
        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None, sharded=True)


    def train_step(self, sess, 
            encoder_inputs, decoder_inputs, 
            targets, target_weights, masks,
            bucket_id, encoder_lens, 
            neAs, Ss, facts, kg_indices,
            forward=False):
    
        batch_size = encoder_inputs[0].shape[0]
        encoder_size, decoder_size = self.buckets[bucket_id]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        input_feed[self.seq_len] = encoder_lens
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.targets[l].name] = targets[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
            input_feed[self.masks[l].name] = masks[l]
        if self.neA != None:
            input_feed[self.neA] = neAs
        if self.S != None:
            input_feed[self.S] = Ss
        if self.facts != None:
            input_feed[self.facts] = facts
        if self.kg_indices != None:
            input_feed[self.kg_indices] = kg_indices

        if forward:
            #output_feed = [self.losses[bucket_id],
            #               self.argmax_outputs[bucket_id],
            #               self.softmax_outputs[bucket_id]]
            output_feed = [self.losses[bucket_id],
                self.argmax_outputs[bucket_id],
                self.softmax_outputs[bucket_id],
                self.a1s[bucket_id], 
                self.kdists[bucket_id], 
                self.Ndists[bucket_id],
                self.Rdebugs[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id],
                           self.op_update[bucket_id]]

        return sess.run(output_feed, input_feed)

    def dynamic_decode(self, sess, encoder_inputs, encoder_lens, decoder_inputs, neAs, Ss, facts, kg_indices, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        input_feed[self.seq_len] = encoder_lens
        input_feed[self.decoder_inputs[0].name] = decoder_inputs[0]
        if self.neA != None:
            input_feed[self.neA] = neAs
        if self.S != None:
            input_feed[self.S] = Ss
        if self.facts != None:
            input_feed[self.facts] = facts
        if self.kg_indices != None:
            input_feed[self.kg_indices] = kg_indices

        output_feed = [self.argmax_outputs[bucket_id],
            self.enc_state[bucket_id],
            self.a1s[bucket_id], 
            self.kdists[bucket_id], 
            self.Ndists[bucket_id],
            self.logits[bucket_id], 
            self.Rdebugs[bucket_id]]

        return sess.run(output_feed, input_feed)
