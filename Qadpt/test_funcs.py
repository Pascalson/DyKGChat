import os
import sys
import tensorflow as tf
import math
import code
import copy
import collections
import queue
import csv
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams

import info_data_utils as data_utils
from compute_knowledge import *
from write_kb_funcs import *
import args
FLAGS = args.FLAGS

KG = data_utils.KG
edim = len(data_utils.edge_types)


# FIXME: neA: nexn -> nxexn, Tproj: nxne->nx1xe
def relax_global_path(Tproj, neA, S, target, loop_id, pre=None, maxlen=6):
    S_indices = np.nonzero(S)[0]
    nt = data_utils.str_nodes.index(target)
    ne_arr = np.take(neA, indices=nt, axis=2)
    ne_indices = np.nonzero(ne_arr)
    paths_lists = []

    if nt in S_indices:
        self_prob = Tproj[nt][0][edim]/np.sum(Tproj[nt])
        paths_lists.append(([target], self_prob))
        
    for ne in zip(ne_indices[0], ne_indices[1]):
        ns_idx = ne[0]
        et = ne[1]
        if pre == ns_idx:
            continue
        if ns_idx == nt:
            continue

        ne_prob = Tproj[ns_idx][0][et] / np.sum(Tproj[ns_idx])
        ns_name = data_utils.str_nodes[ns_idx]
        if et == edim:
            e_type = 'ToSelf'
        else:
            e_type = data_utils.edge_types[et]
        if loop_id < maxlen:
            past_path_lists = relax_global_path(Tproj, neA, S, ns_name, loop_id+1, nt, maxlen)
            for past_path in past_path_lists:
                paths_lists.append(([target, e_type]+past_path[0], ne_prob*past_path[1]))
    return paths_lists


def get_global_path(Tproj, neA, S, target):
    paths_lists = relax_global_path(Tproj, neA, S, target, 0, maxlen=6)
    maxpath = []
    maxprob = 0.
    for path in paths_lists:
        #print('-'.join(path[0]))
        #final = '-'.join(path[0])
        #fpath.write('{},{} |'.format(final, path[1]))
        if path[1] > maxprob:
            maxprob = path[1]
            maxpath = path[0]
    #final = '-'.join(maxpath)
    #fpath.write('{},{}'.format(final, maxprob))
    return maxpath, maxprob

def write_global_path(maxpath, fpath):
    for j in range(round((len(maxpath)-1)/2)):
        fpath.write('{} {} {}'.format(maxpath[2*j+2], maxpath[2*j+1], maxpath[2*j]))
        if j+2 < len(maxpath)-1:
            fpath.write(' < ')


def compute_acc(sess, model, vocabs, data, batch_size=1000, feed_prev=True):

    if feed_prev != False and feed_prev != True:
        raise ValueError("feed_prev must be either True or False.")

    vocab, rev_vocab = vocabs

    encoder_inputs, decoder_inputs, targets, weights, \
        masks, seq_lens, neAs, Ss, facts = data

    print(np.shape(facts[0]))
    '''
    with open('fresults/change_graph_sub_Test1.txt','r') as fng:
        new_graph = fng.readlines()
        for i, new in enumerate(new_graph):
            new = new.strip()
            if new != '':
                if '|' in new:
                    new = new.split('|')
                else:
                    new = [new]
                for n in new:
                    n = n.strip()
                    ns, et, nt = n.split()
                    ns = data_utils.str_nodes.index(ns)
                    et = data_utils.edge_types.index(et)
                    nt = data_utils.str_nodes.index(nt)
                    ne = ns * (edim+1) + et
                    neAs[i][ne] = np.zeros((len(data_utils.str_nodes)))
                    neAs[i][ne][nt] = 1.
                    print('{}: {}-{}-{}'.format(n, ns, et, nt))
    '''
    # RUN
    if feed_prev == False:#testtype=eval_pred_acc
        decoder_inputs = decoder_inputs
    elif feed_prev == True:#testtype=pred_acc
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


    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)
    if not os.path.exists(FLAGS.results_dir+'/Tproj'):
        os.makedirs(FLAGS.results_dir+'/Tproj')

    with open(FLAGS.results_dir+'/outputs.txt','a') as fout, \
        open(FLAGS.results_dir+'/paths.txt','a') as fpath, \
        open(FLAGS.results_dir+'/sources.txt','a') as fS, \
        open(FLAGS.results_dir+'/kws.txt','a') as fkw:
        # ACC
        if feed_prev == False:
            acc_nu = 0.
            TP_num, FN_num, TN_num, FP_num = 0., 0., 0., 0.
            ppx = 0.
        else:
            acc, prec, count, pcount = 0.0, 0.0, 0, 0
            f1 = 0.0
            sen_bleu, corpus_bleu = 0.0, 0.0
            corpus_gts = [[target_ids[i] for target_ids in targets] for i in range(batch_size)]
            kb_counts = {}
            sssps_lens = {}
            distinct_1, distinct_2, distinct_3, distinct_4 = [], [], [], []
            total_word_num = 0

        for i in range(batch_size):
            
            subouts = [output_ids[i] for output_ids in outputs]
            if feed_prev == True:
                if data_utils.EOS_ID in subouts:
                    subouts = subouts[:subouts.index(data_utils.EOS_ID)]
            fout.write(" ".join([tf.compat.as_str(rev_vocab[out]) for out in subouts]))
            fout.write('\n')
            print_S(Ss[i], fS)
            fS.write('\n')

            a1_list = [a1[i] for a1 in a1s]
            kdist_list = [kdist[i] for kdist in kdists]
            Ndist_list = [Ndist[i] for Ndist in Ndists]

            gts = [target_ids[i] for target_ids in targets]
            if data_utils.EOS_ID in gts:
                gts = gts[:gts.index(data_utils.EOS_ID)]

            if feed_prev == False:
                sen_ppx = 0.
                for j, token in enumerate(gts):
                    if token < len(data_utils.str_nodes):
                        if np.argmax(kdist_list[j]) == token:
                            acc_nu += 1
                        if subouts[j] < len(data_utils.str_nodes):
                            TP_num += 1
                        else:
                            FN_num += 1
                        #print('{} : {}, {}'.format(tf.compat.as_str(rev_vocab[token]), \
                        #                    tf.compat.as_str(rev_vocab[np.argmax(kdist_list[j])]), a1_list[j]))#, (copy_list[j] == 1)))
                        #print('-- {}'.format(
                        #    [tf.compat.as_str(rev_vocab[idx]) for idx in np.argwhere(kdist_list[j] == np.max(kdist_list[j])).flatten().tolist()]))
                        sen_ppx += np.log2(a1_list[j] * kdist_list[j][token] + 1e-12)
                        #get_global_path(Rdebugs[j][1][i], neAs[i], Ss[i], data_utils.str_nodes[token], fpath)
                        #fpath.write('  |  ')
                        # TODO
                        #sen_ppx += np.log2(kdist_list[j][token] + 1e-12)
                        '''
                        if a1_list[j] > 0.5:
                        #if copy_list[j] == 1:
                            sen_ppx += np.log2(kdist_list[j][token] + 1e-12)
                        else:
                            sen_ppx += np.log2(1e-12)
                        '''
                    else:
                        if subouts[j] < len(data_utils.str_nodes):
                            FP_num += 1
                        else:
                            TN_num += 1
                        #sen_ppx += np.log2((1-a1_list[j]) * Ndist_list[j][token-len(data_utils.str_nodes)] + 1e-12)
                        sen_ppx += np.log2(Ndist_list[j][token-len(data_utils.str_nodes)] + 1e-12)
                        # TODO
                        #sen_ppx += np.log2(Ndist_list[j][token-len(data_utils.str_nodes)] + 1e-12)
                        '''
                        if a1_list[j] <= 0.5:
                        #if copy_list[j] == 0:
                            sen_ppx += np.log2(Ndist_list[j][token-len(data_utils.str_nodes)] + 1e-12)
                        else:
                            sen_ppx += np.log2(1e-12)
                        '''
                sen_ppx /= len(gts)
                #print(sen_ppx)
                ppx += sen_ppx#2**(-sen_ppx)

            else:
                for ng in list(ngrams(subouts, 1)):
                    if ng not in distinct_1:
                        distinct_1.append(ng)
                for ng in list(ngrams(subouts, 2)):
                    if ng not in distinct_2:
                        distinct_2.append(ng)
                for ng in list(ngrams(subouts, 3)):
                    if ng not in distinct_3:
                        distinct_3.append(ng)
                for ng in list(ngrams(subouts, 4)):
                    if ng not in distinct_4:
                        distinct_4.append(ng)
                total_word_num += len(subouts)
                gt_kws = []
                out_kws = []
                sen_acc = 0.0
                sen_prec = 0.0
                sen_bleu += sentence_bleu([gts], subouts, weights=[0.5,0.5])
                corpus_bleu += sentence_bleu(corpus_gts, subouts, weights=[0.5,0.5])
                for j, token in enumerate(gts):
                    if token < len(data_utils.str_nodes):
                        if token not in gt_kws:
                            gt_kws.append(token)
                for j, token in enumerate(subouts):
                    if token < len(data_utils.str_nodes):
                        if data_utils.str_nodes[token] not in kb_counts:
                            kb_counts[data_utils.str_nodes[token]] = 0
                        kb_counts[data_utils.str_nodes[token]] += 1
                        path, path_prob = get_global_path(Rdebugs[j][1][i], neAs[i], Ss[i], data_utils.str_nodes[token])
                        hops = (len(path)-1)/2
                        if hops not in sssps_lens:
                            sssps_lens[hops] = 0
                        sssps_lens[hops] += 1
                        write_global_path(path, fpath)
                        fpath.write('  |  ')
                        with open(FLAGS.results_dir+'/Tproj/l'+str(i)+'c'+str(j)+'.csv','w', newline='') as fT:
                            writer = csv.writer(fT)
                            print_Tproj(Rdebugs[j][1][i], writer)
                        if token not in out_kws:
                            out_kws.append(token)
                            if token in gt_kws:
                                sen_acc += 1.
                                sen_prec += 1.
                if len(gt_kws) > 0:
                    sen_acc /= len(gt_kws)
                    acc += sen_acc
                    count += 1
                if len(out_kws) > 0:
                    sen_prec /= len(out_kws)
                    prec += sen_prec
                    pcount += 1
                for kw in out_kws:
                    fkw.write(data_utils.str_nodes[kw])
                    fkw.write(' ')
                fkw.write('\n')
            fpath.write('\n')

    if feed_prev == False:
        return [acc_nu, TP_num, FN_num, TN_num, FP_num, ppx]
    else:
        return [acc, prec, count, pcount, sen_bleu, corpus_bleu, kb_counts, sssps_lens, \
            (distinct_1, distinct_2, distinct_3, distinct_4), total_word_num]
