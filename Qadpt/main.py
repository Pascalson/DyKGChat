import tensorflow as tf
import numpy as np
import pickle
import json
import random
import re
import os
import sys
import time
import math
import code

from seq2seq_model import *
import info_data_utils as data_utils
import compute_knowledge as kb
import args
FLAGS = args.FLAGS
_buckets = args._buckets

if FLAGS.test_type == 'realtime_argmax':
    _buckets = [_buckets[-3]]
elif FLAGS.test_type == 'data_argmax' \
    or FLAGS.test_type == 'check_Qadpt' \
    or FLAGS.test_type == 'pred_acc' \
    or FLAGS.test_type == 'eval_pred_acc' \
    or FLAGS.test_type == 'ifchange':
    _buckets = [_buckets[-1]]
else:
    _buckets = _buckets

from inference import *
from test_funcs import *
from change_test_funcs import *
KG = data_utils.KG

###############################

# TRAINING PROCEDURE

###############################

def train():

    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    def build_summaries(): 
        train_loss = tf.Variable(0.)
        tf.summary.scalar("train_loss", train_loss)
        eval_losses = []
        for ids, _ in enumerate(_buckets):
            eval_losses.append(tf.Variable(0.))
            tf.summary.scalar("eval_loss_{}".format(ids), eval_losses[-1])
        summary_vars = [train_loss] + eval_losses
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    # load in data
    train, dev, vocab_path = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
    train_info, dev_info = data_utils.prepare_info(FLAGS.data_dir, FLAGS.data_path)
    train_kb, dev_kb = data_utils.prepare_kb(FLAGS.data_dir, FLAGS.data_path)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    
    #'''
    with tf.Session() as sess:
        
        
        # build the model
        print('[Step] building model...')
        
        model = Qadpt_Seq2Seq(
            'TRAIN',
            data_utils.kdim,
            data_utils.edim+1,
            data_utils.kbembed_size,
            data_utils.triple_num,
            FLAGS.size,
            FLAGS.num_layers,
            len(rev_vocab),
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            feed_prev=False,
            batch_size=FLAGS.batch_size,
            dtype=tf.float32)

        
        # build summary
        # initialization
        
        summary_ops, summary_vars = build_summaries()
        sess.run(tf.variables_initializer(tf.global_variables()))

        log_dir = os.path.join(FLAGS.model_dir, 'log')
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        

        # restore checkpoint

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(ckpt.model_checkpoint_path))
            model.saver.restore(sess, ckpt.model_checkpoint_path)

        gate_model_dir = os.path.join(FLAGS.model_dir, 'ptr_gate')
        ptr_ckpt = tf.train.get_checkpoint_state(gate_model_dir)
        if ptr_ckpt and tf.train.checkpoint_exists(ptr_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(ptr_ckpt.model_checkpoint_path))
            model.ptr_saver.restore(sess, ptr_ckpt.model_checkpoint_path)

        # prepare data
        # TODO-1 make neA (ne X n matrix) and S (n X 1 vector) first
        print('[Step] read in data...')
        
        train_set = read_info_data_with_buckets(train, train_info, train_kb, FLAGS.max_train_data_size, only_kb=True)
        dev_set = read_info_data_with_buckets(dev, dev_info, dev_kb)
        train_buckets_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        print ('each buckets has: {d}'.format(d=train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]

        step_time, loss = 0.0, 0.0
        TW_grad, Tb_grad = 0.0, 0.0
        current_step = 0
        previous_losses = []

        print('[Step] start training...')
        while True:
            
            # select batch data
            #print('[Step] prepare batch...')

            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, targets, weights, masks, seq_lens, neAs, Ss, sub_facts = \
                get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)


            # training step
            #print('[Step] start training...')

            start_time = time.time()

            step_loss, step_TW_grad, step_Tb_grad, _ \
                = model.train_step(sess, encoder_inputs, \
                                   decoder_inputs, targets, weights, masks, \
                                   bucket_id, seq_lens, neAs, Ss, sub_facts)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            
            loss += step_loss / FLAGS.steps_per_checkpoint
            TW_grad += step_TW_grad / FLAGS.steps_per_checkpoint
            Tb_grad += step_Tb_grad / FLAGS.steps_per_checkpoint
            current_step += 1
            

            # print current training progress

            if current_step % FLAGS.steps_per_checkpoint == 0:

                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print("global step %d; learning rate %.4f;"
                      "step-time %.2f; perplexity %.2f; loss %.2f\n"
                      % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity, loss))
                print("TW grad {}; Tb grad {}".format(TW_grad, Tb_grad))

                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.op_lr_decay)
                previous_losses.append(loss)


                # eval
                
                eval_losses = []

                for bucket_id in range(len(_buckets)):
                    
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        eval_losses.append(0.)
                        continue
                    
                    # select batch data from dev

                    encoder_inputs, decoder_inputs, targets, weights, masks, seq_lens, neAs, Ss, sub_facts = \
                        get_batch_with_buckets(dev_set, FLAGS.batch_size, bucket_id)

                    # eval step
                    
                    eval_loss, outputs, _, _, _, _ \
                        = model.train_step(sess, encoder_inputs, \
                                           decoder_inputs, targets, weights, masks, \
                                           bucket_id, seq_lens, neAs, Ss, sub_facts, forward=True)
                    
                    # print eval summary

                    eval_losses.append(eval_loss)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("  eval: bucket %d perplexity %.2f; loss %.2f"
                            % (bucket_id, eval_ppx, eval_loss))
                    

                # write summary

                feed_dict = {}
                feed_dict[summary_vars[0]] = loss
                for ids, key in enumerate(summary_vars[1:1+len(_buckets)]):
                    feed_dict[key] = eval_losses[ids]
                summary_str = sess.run(summary_ops,
                                       feed_dict=feed_dict)
                writer.add_summary(summary_str, model.global_step.eval())
                writer.flush()


                # save checkpoint
                # reset timer and loss
                
                ckpt_path = os.path.join(FLAGS.model_dir, "ckpt")
                model.saver.save(sess, ckpt_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                sys.stdout.flush()
    #'''



###################################

# EVALUATION (w/ ground-truth inputs)

###################################

def evaluate():
        
    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    
    with tf.Session() as sess:

        model = Qadpt_Seq2Seq(
            'TRAIN',
            data_utils.kdim,
            data_utils.edim+1,
            data_utils.kbembed_size,
            data_utils.triple_num,
            FLAGS.size,
            FLAGS.num_layers,
            len(rev_vocab),
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            feed_prev=False,
            batch_size=FLAGS.batch_size,
            dtype=tf.float32)

        #'''
        sess.run(tf.variables_initializer(tf.global_variables()))
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))


        if FLAGS.test_type == 'eval_pred_acc': 
            # TODO: using train/dev data
            #train, dev, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
            #train_kb, dev_kb = data_utils.prepare_kb(FLAGS.data_dir, FLAGS.data_path)
            #test_set = read_info_data_with_buckets(dev, dev_info, FLAGS.max_train_data_size)
            #test_set = read_info_data_with_buckets(train, train_info, 125)
            # TODO: using test data
            test, test_info, test_kb = data_utils.test_info_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
            test_set = read_info_data_with_buckets(test, test_info, test_kb, FLAGS.max_train_data_size)
            acc_nu, acc_de, tp_num, precision_de, ppx = 0, 0, 0, 0, 0
            for i in range(math.floor(len(test_set[-1])/FLAGS.batch_size)):
                data = get_batch_with_buckets(test_set, FLAGS.batch_size, -1, ifrand=False, idx=i)
                values = compute_acc(sess, model, vocabs=(vocab, rev_vocab), data=data, batch_size=FLAGS.batch_size, feed_prev=False)
                acc_nu += values[0]
                acc_de += values[1] + values[2]
                tp_num += values[1]
                precision_de += values[1] + values[4]
                ppx += values[-1]
            print('True Positive = {}'.format(tp_num))
            print('True Positive + False Negative = {}'.format(acc_de))
            print('True Positive + False Positive = {}'.format(precision_de))
            print('correct kb entity prediction = {}'.format(acc_nu))
            print('KW_ACC = {}'.format(acc_nu/acc_de))
            print('Recall = {}'.format(tp_num/acc_de))
            print('Precision = {}'.format(tp_num/precision_de))
            print('F1 = {}'.format(2/(1/(tp_num/acc_de)+1/(tp_num/precision_de))))
            print('Avg Perplexity = {}'.format(ppx/FLAGS.batch_size/(i+1)))


        elif FLAGS.test_type == 'eval':
            train, dev, vocab_path = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
            train_info, dev_info = data_utils.prepare_info(FLAGS.data_dir, FLAGS.data_path)
            train_kb, dev_kb = data_utils.prepare_kb(FLAGS.data_dir, FLAGS.data_path)

            eval_set = read_info_data_with_buckets(train, train_info, train_kb, FLAGS.max_train_data_size)
            #eval_set = read_info_data_with_buckets(dev, dev_info, dev_kb)
            eval_buckets_sizes = [len(eval_set[b]) for b in range(len(_buckets))]
            eval_total_size = float(sum(eval_buckets_sizes))
            print ('each buckets has: {d}'.format(d=eval_buckets_sizes))
            eval_buckets_scale = [sum(eval_buckets_sizes[:i + 1]) / eval_total_size
                                   for i in range(len(eval_buckets_sizes))]

            inference(sess, model, vocabs=(vocab, rev_vocab), data=(eval_set, eval_buckets_scale), batch_size=FLAGS.batch_size, feed_prev=False)


###################################

# INFERENCE PROCEDURE

###################################

def test():
    
        
    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    
    with tf.Session() as sess:

        #model = None
        #"""
        model = Qadpt_Seq2Seq(
            'TEST',
            data_utils.kdim,
            data_utils.edim+1,
            data_utils.kbembed_size,
            data_utils.triple_num,
            FLAGS.size,
            FLAGS.num_layers,
            len(rev_vocab),
            _buckets,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            feed_prev=True,
            batch_size=FLAGS.batch_size,
            dtype=tf.float32)

        sess.run(tf.variables_initializer(tf.global_variables()))
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))

        gate_model_dir = os.path.join(FLAGS.model_dir, 'ptr_gate')
        ptr_ckpt = tf.train.get_checkpoint_state(gate_model_dir)
        if ptr_ckpt and tf.train.checkpoint_exists(ptr_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(ptr_ckpt.model_checkpoint_path))
            model.ptr_saver.restore(sess, ptr_ckpt.model_checkpoint_path)
        #"""

        if FLAGS.test_type == 'pred_acc':
            # TODO: using train/dev data
            #train, dev, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
            #train_info, dev_info = data_utils.prepare_info(FLAGS.data_dir, FLAGS.data_path)
            #train_kb, dev_kb = data_utils.prepare_kb(FLAGS.data_dir, FLAGS.data_path)
            #test_set = read_info_data_with_buckets(dev, dev_info, dev_kb, FLAGS.max_train_data_size)
            #test_set = read_info_data_with_buckets(train, train_info, train_kb, FLAGS.max_train_data_size)
            # TODO: using test data
            test, test_info, test_kb = data_utils.test_info_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
            test_set = read_info_data_with_buckets(test, test_info, test_kb, FLAGS.max_train_data_size)
            accs, precs, counts, pcounts = 0, 0, 0, 0
            sentence_bleus, corpus_bleus = 0.0, 0.0
            kb_counts, sssps_lens = {}, {}
            distincts = [[] for _ in range(4)]
            total_word_num = 0
            for i in range(math.floor(len(test_set[-1])/FLAGS.batch_size)):
                print(i)
                data = get_batch_with_buckets(test_set, FLAGS.batch_size, -1, ifrand=False, idx=i)
                values = compute_acc(sess, model, vocabs=(vocab, rev_vocab), data=data, batch_size=FLAGS.batch_size)
                accs += values[0]
                precs += values[1]
                counts += values[2]
                pcounts += values[3]
                sentence_bleus += values[4]
                corpus_bleus += values[5]
                for key, value in values[6].items():
                    if key not in kb_counts:
                        kb_counts[key] = value
                    else:
                        kb_counts[key] += value
                for key, value in values[7].items():
                    if key not in sssps_lens:
                        sssps_lens[key] = value
                    else:
                        sssps_lens[key] += value
                for k in range(4):
                    distincts[k] = list(set(values[8][k]) | set(distincts[k]))
                total_word_num += values[9]
            print('Avg Recall per sentence = {}/{} = {}'.format(accs, counts, accs/counts))
            print('Avg Precision per sentence = {}/{} = {}'.format(precs, pcounts, precs/pcounts))
            print('Avg F1 per sentence = {}'.format(2/(1/(precs/pcounts)+1/(accs/counts))))#TODO: not the true F1
            print('sentence-level BLEU-2 = {}'.format(sentence_bleus/FLAGS.batch_size/(i+1)))
            print('corpus-level BLEU-2 = {}'.format(corpus_bleus/FLAGS.batch_size/(i+1)))
            kb_proportion = {}
            total_kb_count = sum(kb_counts.values())
            print(total_kb_count)
            for key, value in kb_counts.items():
                kb_proportion[str(key)] = float(value) / total_kb_count
            sorted_kb_by_values = sorted(kb_proportion.items(), key=lambda x: x[1])
            sorted_kb_counts = collections.OrderedDict(sorted_kb_by_values)
            with open(FLAGS.results_dir+'/sorted_kb_counts.json','w') as fkb_counts:
                json.dump(sorted_kb_counts, fkb_counts, indent=2, ensure_ascii=False)
            #print(sorted_kb_counts)
            print(sssps_lens)
            for k in range(4):
                print('distinct-{} = {}'.format(k+1, len(distincts[k])/total_word_num))


        elif FLAGS.test_type == 'ifchange':
            with open(FLAGS.results_dir+'/changed_outputs.txt','w'):
                os.utime(FLAGS.results_dir+'/changed_outputs.txt',None)
            test, test_info, test_kb = data_utils.test_info_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
            test_set = read_info_data_with_buckets(test, test_info, test_kb, FLAGS.max_train_data_size)
            max_data_size = FLAGS.batch_size * math.floor(len(test_set[-1])/FLAGS.batch_size)
            test_set, change_num = ifchange_read_info_data_with_buckets(test, test_info, test_kb, max_data_size)
            have_changed, accu_changed = 0, 0
            for i in range(math.floor(len(test_set[-1])/FLAGS.batch_size)):
                print(i)
                data = get_batch_with_buckets(test_set, FLAGS.batch_size, -1, ifrand=False, idx=i)
                values = ifchange(sess, model, vocabs=(vocab, rev_vocab), data=data, batch_size=FLAGS.batch_size)
                have_changed += values[0]
                accu_changed += values[1]
            print('Changed # KB entities in Responses: {}'.format(have_changed))
            print('Accurately changed # KB entities in Responses: {}'.format(accu_changed))
            print('# KB be changed: {}'.format(change_num))
            print('KB changed rate: {}'.format(float(have_changed)/change_num))
            print('KB accurately changed rate: {}'.format(float(accu_changed)/change_num))
                

        elif FLAGS.test_type == 'realtime_argmax':

            inference(sess, model, vocabs=(vocab, rev_vocab), data=None)

        elif FLAGS.test_type == 'data_argmax':

            train, dev, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
            train_info, dev_info = data_utils.prepare_info(FLAGS.data_dir, FLAGS.data_path)
            train_kb, dev_kb = data_utils.prepare_kb(FLAGS.data_dir, FLAGS.data_path)
            eval_set = read_info_data_with_buckets(train, train_info, train_kb, FLAGS.max_train_data_size)
            #eval_set = read_info_data_with_buckets(dev, dev_info, dev_kb)
            eval_buckets_sizes = [len(eval_set[b]) for b in range(len(_buckets))]
            eval_total_size = float(sum(eval_buckets_sizes))
            print ('each buckets has: {d}'.format(d=eval_buckets_sizes))
            eval_buckets_scale = [sum(eval_buckets_sizes[:i + 1]) / eval_total_size
                                   for i in range(len(eval_buckets_sizes))]
            
            inference(sess, model, vocabs=(vocab, rev_vocab), data=(eval_set, eval_buckets_scale), batch_size=FLAGS.batch_size)



###################################

# Utilities

###################################

def ifchange_read_info_data_with_buckets(data_path, info_path, kb_path, max_size=None, level=0):#CL
    buckets = _buckets
    dataset = [[] for _ in buckets]
    with tf.gfile.GFile(data_path, mode='r') as data_file, \
         tf.gfile.GFile(info_path, mode='rb') as info_file, \
         tf.gfile.GFile(kb_path, mode='rb') as kb_file, \
         open(FLAGS.results_dir+'/ifchange_kws.txt','w') as fif, \
         open(FLAGS.results_dir+'/kws.txt','r') as fkw:

        info = pickle.load(info_file)
        kbs = pickle.load(kb_file)
        reduced_kbs = [kb for kb in kbs if len(kb) > 0]
        source = data_file.readline()
        target = data_file.readline()
        kws = fkw.readlines()

        if level == 1:
            edges = KG.get_edges()

        counter = 0
        change_num = 0
        while source and target and \
                (not max_size or counter < max_size):
            
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(data_utils.EOS_ID)

            S = info[counter]
            kw = kws[counter].strip().split()
            kw_ids = [data_utils.str_nodes.index(w) for w in kw]
            fact = kbs[counter]
            if level == -1:
                if len(kw_ids) > 0:
                    fact = random.choice(reduced_kbs)
                    change_num += 1
                if_kws = []
                rand_nos = []
            else:
                rand_ids_list = [[j for j, triple in enumerate(fact) if triple[2] == k] for k in kw_ids]
                rand_nos = [random.choice(rand_ids) for rand_ids in rand_ids_list if len(rand_ids) > 0]

            counter += 1
            benew = False
            fact_mat = []
            new_pairs = []
            delete_pairs = []
            cur_s = []
            for j, triple in enumerate(fact):
                if len(triple) == 3:
                    nt = triple[2]
                    if j in rand_nos and level == 0:
                        nt = random.randrange(len(data_utils.str_nodes))
                        fif.write(data_utils.str_nodes[nt]+' ')
                        delete_pairs.append(triple)
                        new_pairs.append((triple[0], triple[1], nt))
                        benew = True
                    elif j in rand_nos and level == 1:
                        et0 = triple[1]
                        pair0 = random.choice(edges[data_utils.edge_types[et0]])
                        nt = data_utils.str_nodes.index(pair0[0])
                        nt0 = data_utils.str_nodes.index(pair0[1])
                        fif.write(data_utils.str_nodes[nt0]+' ')
                        delete_pairs.append(triple)
                        new_pairs.append((nt, et0, nt0))
                        cur_s.append(triple[0])
                        benew = True
                    elif level == -1:
                        if triple[0] not in if_kws:
                            if_kws.append(triple[0])
                        if triple[2] not in if_kws:
                            if_kws.append(triple[2])

                    if level <= 0:
                        vec = np.concatenate((data_utils.node_dict[data_utils.str_nodes[triple[0]]],
                                              data_utils.edge_type_dict[data_utils.edge_types[triple[1]]],
                                              data_utils.node_dict[data_utils.str_nodes[nt]]))
                        fact_mat.append(vec)
                if len(fact_mat) >= data_utils.triple_num and level <= 0:
                    break

            if benew == True and level >= 0:
                change_num += 1

            if level == -1:
                fif.write(' '.join([data_utils.str_nodes[k] for k in if_kws]))
            fif.write('\n')
            if level >= 0:
                tmp_fact = [f for f in fact if f not in delete_pairs]
                tmp_fact.extend(new_pairs)
                new_fact = []
                s_used = 0
                for f in tmp_fact:
                    if f[2] not in cur_s or s_used >= 1:#FIXME
                        new_fact.append(f)
                    else:
                        new_fact.append((f[0],f[1], new_pairs[cur_s.index(f[2])][0]))
                        s_used += 1
                fact = new_fact
                #print(len(fact))
                if level >= 1:
                    for triple in fact:
                        vec = np.concatenate((data_utils.node_dict[data_utils.str_nodes[triple[0]]],
                                              data_utils.edge_type_dict[data_utils.edge_types[triple[1]]],
                                              data_utils.node_dict[data_utils.str_nodes[triple[2]]]))
                        fact_mat.append(vec)
                        if len(fact_mat) >= data_utils.triple_num:
                            break
            for _ in range(data_utils.triple_num - len(fact_mat)):
                fact_mat.append(np.zeros(data_utils.kbembed_size,))


            stored = 0
            neA = fact
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    dataset[bucket_id].append([ source_ids, target_ids, neA, S, fact_mat ])
                    stored = 1
                    break
            if stored == 0:#truncate the length
                dataset[-1].append([ source_ids[:buckets[-1][0]], target_ids[:buckets[-1][1]], neA, S, fact_mat ])
                    
            # next loop
            source = data_file.readline()
            target = data_file.readline()

    return dataset, change_num


def read_info_data_with_buckets(data_path, info_path, kb_path, max_size=None, only_kb=False):
    buckets = _buckets
    dataset = [[] for _ in buckets]
    with tf.gfile.GFile(data_path, mode='r') as data_file, \
         tf.gfile.GFile(info_path, mode='rb') as info_file, \
         tf.gfile.GFile(kb_path, mode='rb') as kb_file:
        info = pickle.load(info_file)
        kb = pickle.load(kb_file)
        source = data_file.readline()
        target = data_file.readline()
        counter = 0
        while source and target and \
                (not max_size or counter < max_size):
            S = info[counter]
            neA = kb[counter]
            counter += 1
            
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(data_utils.EOS_ID)

            fact_mat = []
            for triple in neA:
                if len(triple) == 3:
                    vec = np.concatenate((data_utils.node_dict[data_utils.str_nodes[triple[0]]],
                                          data_utils.edge_type_dict[data_utils.edge_types[triple[1]]],
                                          data_utils.node_dict[data_utils.str_nodes[triple[2]]]))
                    fact_mat.append(vec)
                if len(fact_mat) >= data_utils.triple_num:
                    break
            for _ in range(data_utils.triple_num - len(fact_mat)):
                fact_mat.append(np.zeros(data_utils.kbembed_size,))

            stored = 0
            if only_kb == False or len(neA) > 0:
                for bucket_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        dataset[bucket_id].append([ source_ids, target_ids, neA, S, fact_mat ])
                        stored = 1
                        break
                if stored == 0:#truncate the length
                    dataset[-1].append([ source_ids[:buckets[-1][0]], target_ids[:buckets[-1][1]], neA, S, fact_mat ])
                    
            # next loop
            source = data_file.readline()
            target = data_file.readline()

    return dataset


def tensor_neA(neA):
    neA_T = np.zeros((data_utils.kdim, data_utils.edim+1, data_utils.kdim))
    for ne in neA:
        neA_T[ne[0]][ne[1]][ne[2]] = 1.
        neA_T[ne[2]][data_utils.edim][ne[2]] = 1.
    #for n in range(data_utils.kdim):
    #    neA_T[n][data_utils.edim][n] = 1.
        #for e in range(data_utils.edim):
        #    if np.sum(neA_T[n][e]) > 0:
        #        neA_T[n][e] = neA_T[n][e] / np.sum(neA_T[n][e])
    return neA_T

def vector_S(S):
    S_V = np.zeros((data_utils.kdim))
    for s in S:
        S_V[s] = 1.
    if len(S) > 0:
        S_V = S_V / np.sum(S_V)
    return S_V

def get_batch_with_buckets(data, batch_size, bucket_id, ifrand=True, idx=0):

    encoder_size, decoder_size = _buckets[bucket_id]
    encoder_inputs, decoder_inputs, seq_len = [], [], []
    neAs, Ss = [], []
    batch_facts = []

    for i in range(batch_size):
        if ifrand == True:
            encoder_input, decoder_input, neA, S, fact_mat = random.choice(data[bucket_id])
        else:
            encoder_input, decoder_input, neA, S, fact_mat = data[bucket_id][i+idx*batch_size]
            
        neA_T = tensor_neA(neA)
        neAs.append(neA_T)
        S_V = vector_S(S)
        Ss.append(S_V)

        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input)) + encoder_pad)
        seq_len.append(len(encoder_input))
        decoder_pad = [data_utils.PAD_ID] * (decoder_size - len(decoder_input))
        decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)
        batch_facts.append(fact_mat)

    batch_encoder_inputs, batch_decoder_inputs, batch_targets = [], [], []
    batch_weights, batch_masks = [], []
    
    for length_idx in range(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))

    for length_idx in range(decoder_size):
        batch_targets.append(
            np.array([decoder_inputs[batch_idx][length_idx + 1]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))

        batch_weight = np.ones(batch_size, dtype = np.float32)
        batch_mask = np.zeros(batch_size, dtype = np.float32)
        batch_decoder_input = np.zeros(batch_size, dtype = np.float32)
        
        for batch_idx in range(batch_size):
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                batch_weight[batch_idx] = 0.0

            if target >= data_utils.kbstart and target <= data_utils.kbend:
                batch_mask[batch_idx] = 1.0

            dec_inp = decoder_inputs[batch_idx][length_idx]
            #FIXME if dec_inp >= data_utils.kbstart and dec_inp <= data_utils.kbend:
            #    batch_decoder_input[batch_idx] = data_utils.KB_ID
            #else:
            #    batch_decoder_input[batch_idx] = dec_inp
            batch_decoder_input[batch_idx] = dec_inp

        batch_decoder_inputs.append(batch_decoder_input)
        batch_weights.append(batch_weight)
        batch_masks.append(batch_mask)


    return batch_encoder_inputs, batch_decoder_inputs, \
            batch_targets, batch_weights, batch_masks, \
            seq_len, neAs, Ss, batch_facts



###################################

# MAIN

###################################

if __name__ == '__main__':
    if FLAGS.test_type == 'None':
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        with open('{}/model.conf'.format(FLAGS.model_dir),'w') as f:
            for key, value in vars(FLAGS).items():
                f.write("{}={}\n".format(key, value))
        train()
    elif FLAGS.test_type == 'eval' or FLAGS.test_type == 'eval_pred_acc':
        evaluate()
    else:
        test()
