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
import collections
import importlib

from main_model import main_model
from test_funcs import create_out_results, create_path_results, compute_acc
from change_test_funcs import ifchange
from batch_utils import *
import data_utils
import args
FLAGS = args.FLAGS
_buckets = args._buckets

if FLAGS.test_type == 'train':
    _buckets = _buckets
else:# evaluate, test
    _buckets = [_buckets[-1]]

sys.path.insert(1, '/models')

###############################

# TRAINING PROCEDURE

###############################

def train(vocab_path, vocab, rev_vocab, config, more_conf):

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

    print('enter train func')
    print(vocab_path)
    print(config)
    print(more_conf)
    # load in data
    train, dev, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
    train_info, dev_info = data_utils.prepare_info(FLAGS.data_dir, FLAGS.data_path)
    train_kb, dev_kb = data_utils.prepare_kb(FLAGS.data_dir, FLAGS.data_path)
    
    with tf.Session() as sess:
        
        
        # build the model
        print('[Step] building model...')
        model = main_model(
            'TRAIN', *config, **more_conf)
        
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

        # prepare data
        print('[Step] read in data...')
        
        train_set = read_info_data_with_buckets(train, train_info, train_kb, FLAGS.max_train_data_size, only_kb=False)
        dev_set = read_info_data_with_buckets(dev, dev_info, dev_kb)
        train_buckets_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        print ('each buckets has: {d}'.format(d=train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        print('[Step] start training...')
        while True:
            
            # prepare batch
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, targets, weights, masks, seq_lens, neAs, Ss, sub_facts, kg_indices = \
                get_batch_with_buckets(train_set, FLAGS.batch_size, bucket_id)


            # training step
            start_time = time.time()
            step_loss, _ \
                = model.train_step(sess, encoder_inputs, \
                                   decoder_inputs, targets, weights, masks, \
                                   bucket_id, seq_lens, neAs, Ss, sub_facts, kg_indices)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # print progress
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print("global step %d; learning rate %.4f;"
                      "step-time %.2f; perplexity %.2f; loss %.2f\n"
                      % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity, loss))
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
                    
                    # prepare dev batch
                    encoder_inputs, decoder_inputs, targets, weights, masks, seq_lens, neAs, Ss, sub_facts, kg_indices = \
                        get_batch_with_buckets(dev_set, FLAGS.batch_size, bucket_id)

                    # eval step
                    eval_loss, outputs, _, _, _, _, _ \
                        = model.train_step(sess, encoder_inputs, \
                                           decoder_inputs, targets, weights, masks, \
                                           bucket_id, seq_lens, neAs, Ss, sub_facts, kg_indices, forward=True)
                    
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



###################################

# EVALUATION (w/ ground-truth inputs)

###################################

def evaluate(vocab_path, vocab, rev_vocab, config, more_conf):
    print("enter evaluate func")
    with tf.Session() as sess:
        # build the model
        print('[Step] building model...')
        model = main_model(
            'TRAIN', *config, **more_conf)

        # initialize and store back parameters
        sess.run(tf.variables_initializer(tf.global_variables()))
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))

        if FLAGS.test_type == 'eval_pred_acc': 
            outs_dir = create_out_results('eval')
            results_dirs = [outs_dir]
            # using train/dev data
            if FLAGS.data_type == 'train' or FLAGS.data_type == 'dev':
                train, dev, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
                train_kb, dev_kb = data_utils.prepare_kb(FLAGS.data_dir, FLAGS.data_path)
                if FLAGS.data_type == 'dev':
                    test_set = read_info_data_with_buckets(dev, dev_info, FLAGS.max_train_data_size)
                else:
                    # we do not test the whole training data
                    # but only randomly select a batch.
                    test_set = read_info_data_with_buckets(train, train_info, FLAGS.batch_size)

            # using test data
            else:
                test, test_info, test_kb = data_utils.test_info_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
                test_set = read_info_data_with_buckets(test, test_info, test_kb, FLAGS.max_train_data_size)

            acc_nu, acc_de, tp_num, precision_de, ppx = 0, 0, 0, 0, 0
            for i in range(math.floor(len(test_set[-1])/FLAGS.batch_size)):
                data = get_batch_with_buckets(test_set, FLAGS.batch_size, -1, ifrand=False, idx=i)
                values = compute_acc(sess, results_dirs, model, vocabs=(vocab, rev_vocab), data=data, batch_size=FLAGS.batch_size, feed_prev=False)
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
            print('Avg Perplexity = {}'.format(2**(-ppx/FLAGS.batch_size/(i+1))))



###################################

# INFERENCE PROCEDURE

###################################

def test(vocab_path, vocab, rev_vocab, config, more_conf):
    print("enter test func")
    with tf.Session() as sess:
        # build the model
        print('[Step] building model...')
        model = main_model(
            'TEST', *config, **more_conf)

        sess.run(tf.variables_initializer(tf.global_variables()))
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))

        gate_model_dir = os.path.join(FLAGS.model_dir, 'ptr_gate')
        ptr_ckpt = tf.train.get_checkpoint_state(gate_model_dir)
        if ptr_ckpt and tf.train.checkpoint_exists(ptr_ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(ptr_ckpt.model_checkpoint_path))
            model.ptr_saver.restore(sess, ptr_ckpt.model_checkpoint_path)

        if FLAGS.test_type == 'pred_acc':
            outs_dir = create_out_results('test')
            paths_dir, kws_dir = create_path_results('test')
            results_dirs = [outs_dir, paths_dir, kws_dir]
            # using train/dev data
            if FLAGS.data_type == 'train' or FLAGS.data_type == 'dev':
                train, dev, _ = data_utils.prepare_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
                train_info, dev_info = data_utils.prepare_info(FLAGS.data_dir, FLAGS.data_path)
                train_kb, dev_kb = data_utils.prepare_kb(FLAGS.data_dir, FLAGS.data_path)
                if FLAGS.data_type == 'dev':
                    test_set = read_info_data_with_buckets(dev, dev_info, dev_kb, FLAGS.max_train_data_size)
                else:
                    test_set = read_info_data_with_buckets(train, train_info, train_kb, FLAGS.batch_size)
            # using test data
            else:
                test, test_info, test_kb = data_utils.test_info_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
                test_set = read_info_data_with_buckets(test, test_info, test_kb, FLAGS.max_train_data_size)
            print(FLAGS.data_type)
            print(test_set[0][0][0])
            print(len(test_set))
            print(len(test_set[0]))

            accs, precs, counts, pcounts = 0, 0, 0, 0
            sentence_bleus = 0.0
            kb_counts, sssps_lens = {}, {}
            distincts = [[] for _ in range(4)]
            total_word_num = 0
            for i in range(math.floor(len(test_set[-1])/FLAGS.batch_size)):
                print(i)
                data = get_batch_with_buckets(test_set, FLAGS.batch_size, -1, ifrand=False, idx=i)
                values = compute_acc(sess, results_dirs, model, vocabs=(vocab, rev_vocab), data=data, batch_size=FLAGS.batch_size)
                accs += values[0]
                precs += values[1]
                counts += values[2]
                pcounts += values[3]
                sentence_bleus += values[4]
                for key, value in values[5].items():
                    if key not in kb_counts:
                        kb_counts[key] = value
                    else:
                        kb_counts[key] += value
                for key, value in values[6].items():
                    if key not in sssps_lens:
                        sssps_lens[key] = value
                    else:
                        sssps_lens[key] += value
                for k in range(4):
                    distincts[k] = list(set(values[7][k]) | set(distincts[k]))
                total_word_num += values[8]
            print('Avg Recall per sentence = {}/{} = {}'.format(accs, counts, accs/counts))
            print('Avg Precision per sentence = {}/{} = {}'.format(precs, pcounts, precs/pcounts))
            print('Avg F1 per sentence = {}'.format(2/(1/(precs/pcounts)+1/(accs/counts))))
            print('sentence-level BLEU-2 = {}'.format(sentence_bleus/FLAGS.batch_size/(i+1)))
            kb_proportion = {}
            total_kb_count = sum(kb_counts.values())
            print(total_kb_count)
            for key, value in kb_counts.items():
                kb_proportion[str(key)] = float(value) / total_kb_count
            sorted_kb_by_values = sorted(kb_proportion.items(), key=lambda x: x[1])
            sorted_kb_counts = collections.OrderedDict(sorted_kb_by_values)
            with open(FLAGS.results_dir+'/sorted_kb_counts.json','w') as fkb_counts:
                json.dump(sorted_kb_counts, fkb_counts, indent=2, ensure_ascii=False)
            print(sssps_lens)
            for k in range(4):
                print('distinct-{} = {}'.format(k+1, len(distincts[k])/total_word_num))


        elif FLAGS.test_type == 'ifchange':
            outs_dir = create_out_results('change')
            results_dirs = [outs_dir]
            # create file to store outputs 
            '''
            with open(FLAGS.results_dir+'/changed_outputs.txt','w'):
                os.utime(FLAGS.results_dir+'/changed_outputs.txt',None)
            '''
            test, test_info, test_kb = data_utils.test_info_data(FLAGS.data_dir, FLAGS.data_path, FLAGS.vocab_size)
            test_set = read_info_data_with_buckets(test, test_info, test_kb, FLAGS.max_train_data_size)
            max_data_size = FLAGS.batch_size * math.floor(len(test_set[-1])/FLAGS.batch_size)
            test_set, change_num = ifchange_read_info_data_with_buckets(test, test_info, test_kb, max_data_size, level=FLAGS.change_level)
            have_changed, accu_changed = 0, 0
            for i in range(math.floor(len(test_set[-1])/FLAGS.batch_size)):
                print(i)
                data = get_batch_with_buckets(test_set, FLAGS.batch_size, -1, ifrand=False, idx=i)
                values = ifchange(sess, results_dirs, model, vocabs=(vocab, rev_vocab), data=data, batch_size=FLAGS.batch_size)
                have_changed += values[0]
                accu_changed += values[1]
            print('Changed # KB entities in Responses: {}'.format(have_changed))
            print('Accurately changed # KB entities in Responses: {}'.format(accu_changed))
            print('# KB be changed: {}'.format(change_num))
            print('KB changed rate: {}'.format(float(have_changed)/change_num))
            print('KB accurately changed rate: {}'.format(float(accu_changed)/change_num))



###################################

# MAIN

###################################

if __name__ == '__main__':
    # build vocabulary
    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    # basic config
    config = (
        FLAGS.model,
        data_utils.kdim,
        data_utils.edim+1,
        data_utils.kbembed_size,
        data_utils.triple_num,
        FLAGS.size,
        FLAGS.num_layers,
        len(rev_vocab),
        _buckets)

    more_conf = {
        'hops_num': FLAGS.hops_num,
        'kgpath_len': FLAGS.kgpath_len,
        'learning_rate': FLAGS.lr,
        'learning_rate_decay_factor': FLAGS.lr_decay,
        'max_gradient_norm': FLAGS.grad_norm,
        'feed_prev': False,
        'batch_size': FLAGS.batch_size,
        'dtype': tf.float32}

    shared_vars = (vocab_path, vocab, rev_vocab, config, more_conf)

    # three types to build the model
    if FLAGS.test_type == 'train':
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        with open('{}/model.conf'.format(FLAGS.model_dir),'w') as f:
            for key, value in vars(FLAGS).items():
                f.write("{}={}\n".format(key, value))
        print('launch train()')
        train(*shared_vars)
    elif FLAGS.test_type == 'eval_pred_acc':
        evaluate(*shared_vars)
    else:
        test(*shared_vars)

