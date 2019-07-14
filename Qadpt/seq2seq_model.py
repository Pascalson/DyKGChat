import tensorflow as tf
import numpy as np
import copy

import info_data_utils as data_utils
from units import *

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class Qadpt_Seq2Seq:
    def __init__(
            self,
            mode,
            kdim,
            edim,
            kbembed_size,
            triples_num,
            size,
            num_layers,
            vocab_size,
            buckets,
            learning_rate=0.5,
            learning_rate_decay_factor=0.99,
            max_gradient_norm=5.0,
            feed_prev=False,
            batch_size=32,
            dtype=tf.float32):

        self.size = size
        self.num_layers = num_layers
        self.kdim = kdim
        self.edim = edim
        self.kbembed_size = kbembed_size
        self.triples_num = triples_num
        self.k_len = 6

        self.vocab_size = vocab_size
        print('VOCABSIZE:{}'.format(vocab_size))
        #num_sampled = 512

        self.buckets = buckets
        self.feed_prev = feed_prev
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.op_lr_decay = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        

        self.cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        self.enc_cell = core_rnn_cell.EmbeddingWrapper(
            cell=self.enc_cell,
            embedding_classes=vocab_size,
            embedding_size=size)


        # output projection
        w = tf.get_variable('proj_w', [size, vocab_size - self.kdim])
        w_t = tf.transpose(w)
        b = tf.get_variable('proj_b', [vocab_size - self.kdim])
        self.output_projection = (w, b)
        

        # input embedding
        self.embedding = variable_scope.get_variable('embedding', [vocab_size, size])

        # MemNet
        #self.memA = tf.Variable(xavier_init([size*num_layers, kbembed_size]))
        #self.memC = tf.Variable(xavier_init([size*num_layers, kbembed_size]))


        # PtrNet
        #self.ptr_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for _ in range(num_layers)])
        #TODO self.ptr_W = tf.Variable(xavier_init([size, 1]))
        #TODO self.ptr_b = tf.Variable(tf.zeros(shape=[1]))

        # purturb
        #self.purturb_W = tf.Variable(xavier_init([size, self.kdim]))
        #self.purturb_b = tf.Variable(tf.zeros(shape=[self.kdim]))

        # Tpred
        self.Tpred_W = tf.Variable(xavier_init([size, self.kdim*self.edim]))
        self.Tpred_b = tf.Variable(tf.zeros(shape=[self.kdim*self.edim]))

        # knowledge placeholder
        self.S = tf.placeholder(tf.float32, shape = [None, self.kdim])
        self.neA = tf.placeholder(tf.float32, \
            shape = [None, self.kdim, self.edim, self.kdim])
        # knowledge placeholder
        self.facts = tf.placeholder(tf.float32, \
            shape = [None, triples_num, kbembed_size])

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


        def MemNet(enc_state):
            query = tf.reshape(enc_state, [-1, 1, size*num_layers])
            for _ in range(self.k_len):
                M = tf.tensordot(self.facts, tf.transpose(self.memA), [[2],[0]])# ? x n x d
                M = tf.transpose(M, [0, 2, 1])
                P = tf.matmul(query, M)# ? x 1 x n
                P = tf.nn.softmax(P)
                C = tf.tensordot(self.facts, tf.transpose(self.memC), [[2],[0]])# ? x n x d
                mem_out = tf.matmul(P, C)
                query = query + mem_out
            return tf.reshape(query, [-1, size*num_layers])


        def reasoning(logit):
            # Test: neA(normed), S(normed)
            # Test_norm: neA(w/o normed), S(normed) -> success
            T = tf.matmul(logit, self.Tpred_W) + self.Tpred_b
            Tproj = tf.reshape(T, [-1, self.kdim, 1, self.edim])
            Tproj = tf.nn.softmax(Tproj, axis=-1)
            Tbsize = tf.shape(T)[0]

            R = tf.squeeze(tf.matmul(Tproj, self.neA),[2])# ? x kdim x 1 x kdim -> ? x kdim x kdim
            R = R / (tf.reduce_sum(R, 2, keep_dims=True)+1e-12)#TODO-Test

            kdists = [tf.reshape(self.S, [-1, 1, self.kdim])]
            for _ in range(self.k_len-1):
                kdists.append(tf.matmul(kdists[-1], R))

            kdist = tf.squeeze(kdists[-1], [1])
            kdist = kdist / (tf.reduce_sum(kdist, 1, keep_dims=True)+1e-12)#TODO-Test
            #purturb = tf.clip_by_value(tf.matmul(logit, self.purturb_W) + self.purturb_b, -0.1, 0.1)
            #kdist += purturb

            return kdist, [T, Tproj, R, kdists]

        #'''
        def PtrGate(logit):
            a1 = tf.nn.sigmoid(tf.matmul(logit, self.ptr_W) + self.ptr_b)
            return a1
        #'''

        def PtrNet(hid_states, logits, ptr_logits):
            outputs, Ndists, kdists, a1s = [], [], [], []
            total_dists = []
            Rdebugs = []
            for hid_state, logit, ptr_logit in zip(hid_states, logits, ptr_logits):
                kdist, Rdebug = reasoning(logit)
                Ndist = tf.nn.softmax(nn_ops.xw_plus_b(
                            logit, self.output_projection[0], self.output_projection[1]))
                #a1 = PtrGate(ptr_logit)
                a1 = tf.reshape(tf.gather(Ndist, tf.constant(4), axis=1), [-1, 1])#tf.slice(Ndist,[0,4],[1,-1])
                Ndist1 = tf.gather(Ndist, list(range(4)), axis=1)
                Ndist2 = tf.zeros(tf.shape(a1))
                Ndist3 = tf.gather(Ndist, list(range(5, self.vocab_size - self.kdim)), axis=1)
                Ndist = tf.concat([Ndist1, Ndist2, Ndist3], 1)
                '''
                out_symbol = tf.where(tf.reshape(a1, [-1]) > 0.5,
                                math_ops.argmax(kdist, axis=1),
                                math_ops.argmax(Ndist, axis=1) + self.kdim)
                '''
                #total_dist = [[k[i] + N[i] for i in range(self.kdim)] + N[self.kdim:] for k, N in zip(kdist, Ndist)]
                #total_dist = tf.concat([a1*kdist, (1-a1)*Ndist],1)
                total_dist = tf.concat([a1*kdist, Ndist],1)
                out_symbol = math_ops.argmax(total_dist, axis=1)

                outputs.append(out_symbol)
                a1s.append(a1)
                kdists.append(kdist)
                Ndists.append(Ndist)
                total_dists.append(total_dist)
                Rdebugs.append(Rdebug)
            return outputs, a1s, kdists, Ndists, total_dists, Rdebugs

        def loop_function(hid_state,logit):
            kdist, Rdebug = reasoning(logit)
            Ndist = tf.nn.softmax(nn_ops.xw_plus_b(
                        logit, self.output_projection[0], self.output_projection[1]))
            #a1 = PtrGate(logit)
            a1 = tf.reshape(tf.gather(Ndist, tf.constant(4), axis=1), [-1, 1])#tf.slice(Ndist,[0,4],[1,-1])
            Ndist1 = tf.gather(Ndist, list(range(4)), axis=1)
            Ndist2 = tf.zeros(tf.shape(a1))
            Ndist3 = tf.gather(Ndist, list(range(5, self.vocab_size - self.kdim)), axis=1)
            Ndist = tf.concat([Ndist1, Ndist2, Ndist3], 1)

            #total_dist = tf.concat([a1*kdist, (1-a1)*Ndist],1)
            total_dist = tf.concat([a1*kdist, Ndist],1)
            out_symbol = math_ops.argmax(total_dist, axis=1)
            a1 = tf.reshape(a1, [-1])
            '''
            out_symbol = tf.where(a1 > 0.5,
                            math_ops.argmax(kdist, axis=1),
                            math_ops.argmax(Ndist, axis=1) + self.kdim)
            '''
            #FIXME kb_embeddings = tf.tile([embedding_ops.embedding_lookup(self.embedding, data_utils.KB_ID)], [tf.shape(logit)[0], 1])
            '''
            emb_prev = tf.where(a1 > 0.5,
                            kb_embeddings,
                            embedding_ops.embedding_lookup(self.embedding, out_symbol))
            '''
            #FIXME emb_prev = tf.where(out_symbol < self.kdim,
            #                kb_embeddings,
            #                embedding_ops.embedding_lookup(self.embedding, out_symbol))
            emb_prev = embedding_ops.embedding_lookup(self.embedding, out_symbol)
            return [emb_prev, out_symbol, a1, kdist, Ndist, Rdebug]


        def compute_loss(total_dists, targets, weights):
            
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


        if mode == 'TRAIN':

            self.enc_state = []
            self.losses = []
            self.decKB_losses = []
            self.decN_losses = []
            self.ptr_losses = []
            self.outputs = []
            self.a1s = []
            self.kdists = []
            self.Ndists = []
            self.logits = []
            self.Rdebugs = []

            for j, bucket in enumerate(buckets):

                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True if j > 0 else None):

                    _, enc_state = \
                        encode(self.enc_cell, self.encoder_inputs[:bucket[0]], self.seq_len)

                    #mem_out = MemNet(tf.reshape(enc_state,[-1, self.size*self.num_layers]))
                    #enc_state = tf.tuple([tf.slice(mem_out, [0, l*self.size], [-1, self.size]) for l in range(self.num_layers)])

                    logits, hiddens, dec_state = \
                        decode(self.cell, enc_state, self.embedding, \
                               self.decoder_inputs[:bucket[1]], None, \
                               bucket[1]+1, feed_prev=False)
                    
                    outputs, a1s, kdists, Ndists, total_dists, Rdebug = PtrNet(hiddens, logits, logits)

                    loss = compute_loss(total_dists, self.targets[:bucket[1]], self.target_weights[:bucket[1]])
                    #penalty = tf.reduce_sum(tf.reduce_sum(tf.square(Rdebug[j][1]), axis=[1,2,3]) - self.k_len)
                    #loss = loss + 0.01*penalty

                    self.enc_state.append(enc_state)
                    self.losses.append(loss)
                    self.outputs.append(outputs)
                    self.a1s.append(a1s)
                    self.kdists.append(kdists)
                    self.Ndists.append(Ndists)
                    self.logits.append(logits)
                    self.Rdebugs.append(Rdebug)

            # check
            self.argmax_outputs = self.outputs

            # update methods
            self.op_update = []
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            params = tf.trainable_variables()
            for idx, p in enumerate(params):
                if p == self.Tpred_W:
                    TW_idx = idx
                if p == self.Tpred_b:
                    Tb_idx = idx
            print(params)
            print('W:{}; b:{}'.format(TW_idx, Tb_idx))
            self.TW_grad, self.Tb_grad = [], []
            for j in range(len(self.buckets)):
                gradients = tf.gradients(self.losses[j], params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.op_update.append(optimizer.apply_gradients(zip(clipped_gradients, params),
                                                                global_step=self.global_step))
                self.TW_grad.append(gradients[TW_idx])
                self.Tb_grad.append(gradients[Tb_idx])



        elif mode == 'TEST':
            self.enc_state = []
            self.argmax_outputs = []
            self.a1s = []
            self.kdists = []
            self.Ndists = []
            self.logits = []
            self.Rdebugs = []

            for j, bucket in enumerate(buckets):

                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True if j > 0 else None):

                    _, enc_state = \
                        encode(self.enc_cell, self.encoder_inputs[:bucket[0]], self.seq_len)

                    #mem_out = MemNet(tf.reshape(enc_state,[-1, self.size*self.num_layers]))
                    #enc_state = tf.tuple([tf.slice(mem_out, [0, l*self.size], [-1, self.size]) for l in range(self.num_layers)])

                    logits, argmax_outputs, hiddens, a1s, kdists, Ndists, Rdebugs = \
                        decode(self.cell, enc_state, self.embedding, \
                               self.decoder_inputs[:bucket[1]], None, bucket[1], \
                               feed_prev=True, loop_function=loop_function)

                    #lm_logits, _, _, _, _, _, _ = \
                    #    decode(self.cell, self.cell.zeros_state(local_batch_size, tf.float32), self.embedding, \
                    #           self.decoder_inputs[:bucket[1]], None, bucket[1], \
                    #           feed_prev=False)

                self.enc_state.append(enc_state)
                self.argmax_outputs.append(argmax_outputs)
                self.a1s.append(a1s)
                self.kdists.append(kdists)
                self.Ndists.append(Ndists)
                self.logits.append(logits)
                self.Rdebugs.append(Rdebugs)

            params = tf.trainable_variables()
            print(params)

        # saver
        self.saver = tf.train.Saver(max_to_keep=None, sharded=True)
        #self.no_gate_saver = tf.train.Saver(var_list=[p for p in tf.trainable_variables() if p not in [self.ptr_W, self.ptr_b, self.global_step, self.learning_rate]])
        #self.ptr_saver = tf.train.Saver(var_list=[self.ptr_W, self.ptr_b], max_to_keep=None, sharded=True)



    def train_step(self, sess, encoder_inputs, decoder_inputs, targets, target_weights, masks,
            bucket_id, encoder_lens, neAs, Ss, facts, forward=False):
    
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
        input_feed[self.neA] = neAs
        input_feed[self.S] = Ss
        input_feed[self.facts] = facts

        if forward:
            output_feed = [self.losses[bucket_id],
                           self.argmax_outputs[bucket_id],
                           self.a1s[bucket_id],
                           self.kdists[bucket_id],
                           self.Ndists[bucket_id],
                           self.Rdebugs[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id],
                           self.TW_grad[bucket_id],
                           self.Tb_grad[bucket_id],
                           self.op_update[bucket_id]]

        return sess.run(output_feed, input_feed)

    def dynamic_decode(self, sess, encoder_inputs, encoder_lens, decoder_inputs, neAs, Ss, facts, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        input_feed[self.seq_len] = encoder_lens
        input_feed[self.decoder_inputs[0].name] = decoder_inputs[0]
        input_feed[self.neA] = neAs
        input_feed[self.S] = Ss
        input_feed[self.facts] = facts

        output_feed = [self.argmax_outputs[bucket_id], self.enc_state[bucket_id]
                       , self.a1s[bucket_id], self.kdists[bucket_id], self.Ndists[bucket_id]
                       , self.logits[bucket_id], self.Rdebugs[bucket_id]]

        return sess.run(output_feed, input_feed)

    def stepwise_test_beam(self, sess, encoder_inputs, encoder_lens, decoder_inputs):
        encoder_size, decoder_size = self.buckets[-1]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        input_feed[self.seq_len] = encoder_lens
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        output_feed = [self.outs]
        return sess.run(output_feed, input_feed)

    def lm_prob(self, sess, decoder_inputs):
        _, decoder_size = self.buckets[-1]
        input_feed = {}
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        output_feed = [self.lm_outs]
        return sess.run(output_feed, input_feed)
