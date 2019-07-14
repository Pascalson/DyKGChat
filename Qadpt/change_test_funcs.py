import sys
import tensorflow as tf
import math
import code
import copy
import collections
import queue
import random

import info_data_utils as data_utils
from compute_knowledge import *
import args
FLAGS = args.FLAGS


KG = data_utils.KG
edges = KG.get_edges()
edim = len(data_utils.edge_types)

def ifchange(sess, model, vocabs, data, batch_size=1000, feed_prev=True, level=1):

    if feed_prev != False and feed_prev != True:
        raise ValueError("feed_prev must be either True or False.")
    if level not in [-1, 0, 1]:
        raise ValueError("level must be -1, 0, or 1")

    vocab, rev_vocab = vocabs

    encoder_inputs, decoder_inputs, targets, weights, \
        masks, seq_lens, neAs, Ss, facts = data

    with open(FLAGS.results_dir+'/changed_outputs.txt','r') as fin:
        cur_idx = len(fin.readlines())

    change_num = 0
    '''
    if level == -1:
        random.shuffle(neAs)
        change_num = len(neAs)

    else:
        change_num = 0
        with open(FLAGS.results_dir+'/paths.txt','r') as fp, \
            open(FLAGS.results_dir+'/ifchange_kws.txt','a') as fif:
            paths = fp.readlines()
            paths = paths[cur_idx:(cur_idx+batch_size)]
            for i, all_path in enumerate(paths):
                all_path = all_path.strip()
                if all_path == '':
                    fif.write('\n')
                else:
                    if '|' in all_path:
                        all_path = all_path.split('|')
                    else:
                        all_path = [all_path]
                    benew = False
                    for path in all_path:
                        if path.strip() == 'none.' or path.strip() == '':
                            continue
                        path = path.split('<') 
                        #print(path)
                        for j, p in enumerate(path):
                            p = p.strip()
                            if len(p.split()) == 3:
                                s, et, t = p.split()
                                s = data_utils.str_nodes.index(s)
                                if et == 'ToSelf':
                                    et = edim
                                else:
                                    et = data_utils.edge_types.index(et)
                                t = data_utils.str_nodes.index(t)

                                if j == 0 and level == 0:
                                    #print('{}/{}'.format(i,len(neAs)))
                                    #print('{}/{}'.format(se,len(neAs[i])))
                                    nt = random.randrange(len(data_utils.str_nodes))
                                    fif.write(data_utils.str_nodes[nt]+' ')
                                    print('{}-{}-{}/{}'.format(s, et, t, nt))
                                    print('{}-{}-{}/{}'.format(data_utils.str_nodes[s], data_utils.edge_types[et], data_utils.str_nodes[t], data_utils.str_nodes[nt]))
                                    neAs[i][s][et] = np.zeros((len(data_utils.str_nodes)))
                                    neAs[i][s][et][nt] = 1.
                                    benew = True
                                if j == 1 and level == 1:
                                    str_et0 = path[0].strip().split()[1]
                                    et0 = data_utils.edge_types.index(str_et0)
                                    pair0 = random.choice(edges[str_et0])
                                    nt = data_utils.str_nodes.index(pair0[0])
                                    nt0 = data_utils.str_nodes.index(pair0[1])
                                    fif.write(data_utils.str_nodes[nt0]+' ')

                                    neAs[i][nt][et0] = np.zeros((len(data_utils.str_nodes)))
                                    neAs[i][nt][et0][nt0] = 1.

                                    neAs[i][s][et] = np.zeros((len(data_utils.str_nodes)))
                                    neAs[i][s][et][nt] = 1.

                                    print('{}-{}-{}/{}'.format(s, et, t, nt))
                                    print('{}-{}-{}/{}'.format(data_utils.str_nodes[s], data_utils.edge_types[et], data_utils.str_nodes[t], data_utils.str_nodes[nt]))
                                    benew = True
                    if benew == True:
                        change_num += 1
                    fif.write('\n')
    '''
    #"""
    # RUN
    if feed_prev == False:
        decoder_inputs = decoder_inputs
    elif feed_prev == True:
        decoder_inputs = [[data_utils.GO_ID for _ in range(batch_size)]]


    if feed_prev == False:
        eval_loss, \
                outputs, a1s, kdists, Ndists, Rdebugs \
            = model.train_step(sess, encoder_inputs, \
                               decoder_inputs, targets, weights, masks, \
                               -1, seq_lens, neAs, Ss, facts, forward=True)
    elif feed_prev == True:
        outputs, enc_state, a1s, kdists, Ndists, logits, Rdebugs \
            = model.dynamic_decode(sess, encoder_inputs, seq_lens, \
                                   decoder_inputs, neAs, Ss, facts, -1)

    #'''
    with open(FLAGS.results_dir+'/changed_outputs.txt','a') as fout, \
        open(FLAGS.results_dir+'/outputs.txt','r') as fin, \
        open(FLAGS.results_dir+'/ifchange_kws.txt','r') as fif:

        origins = fin.readlines()
        if_kws = fif.readlines()
        have_changed, accu_changed = 0, 0
        for i in range(batch_size):
            
            subouts = [output_ids[i] for output_ids in outputs]
            if data_utils.EOS_ID in subouts:
                subouts = subouts[:subouts.index(data_utils.EOS_ID)]
            strout = " ".join([tf.compat.as_str(rev_vocab[out]) for out in subouts])
            fout.write(strout)
            fout.write('\n')

            ori = origins[i+cur_idx].strip()

            #for ori_t, cur_t in zip(ori.split(), strout.split()):
            #    if ori_t != cur_t and ori_t in data_utils.str_nodes and cur_t in data_utils.str_nodes:
            #        have_changed += 1

            if strout != ori:
                have_changed += 1
            out_kws = [x for x in strout.split() if x in data_utils.str_nodes]
            for x in if_kws[i+cur_idx].strip().split():
                if x in out_kws:
                    accu_changed += 1
                    break

        #print(have_changed)
        #print(change_num)
        return [have_changed, accu_changed]
