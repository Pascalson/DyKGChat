import tensorflow as tf
import numpy as np
import random
import pickle
import args
FLAGS = args.FLAGS
_buckets = args._buckets
if FLAGS.test_type == 'train':
    _buckets = _buckets
else:# evaluate, test
    _buckets = [_buckets[-1]]

import data_utils
KG = data_utils.KG


def read_info_data_with_buckets(data_path, info_path, kb_path, max_size=None, only_kb=False):
    buckets = _buckets
    print(buckets)
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
            fact_indices = []
            for triple in neA:
                if len(triple) == 3:
                    vec = np.concatenate((data_utils.node_dict[data_utils.str_nodes[triple[0]]],
                                          data_utils.edge_type_dict[data_utils.edge_types[triple[1]]],
                                          data_utils.node_dict[data_utils.str_nodes[triple[2]]]))
                    fact_mat.append(vec)
                    fact_indices.append(triple[2])
                if len(fact_mat) >= data_utils.triple_num:
                    break
            for _ in range(data_utils.triple_num - len(fact_mat)):
                fact_mat.append(np.zeros(data_utils.kbembed_size,))
                fact_indices.append(data_utils.kdim)

            stored = 0
            if only_kb == False or len(neA) > 0:
                for bucket_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        dataset[bucket_id].append([ source_ids, target_ids, neA, S, fact_mat, fact_indices ])
                        stored = 1
                        break
                if stored == 0:#truncate the length
                    dataset[-1].append([ source_ids[:buckets[-1][0]], target_ids[:buckets[-1][1]], neA, S, fact_mat, fact_indices ])
                    
            # next loop
            source = data_file.readline()
            target = data_file.readline()

    return dataset


def tensor_neA(neA):
    neA_T = np.zeros((data_utils.kdim, data_utils.edim+1, data_utils.kdim))
    for ne in neA:
        neA_T[ne[0]][ne[1]][ne[2]] = 1.
        neA_T[ne[2]][data_utils.edim][ne[2]] = 1.
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
    batch_kg_indices = []

    for i in range(batch_size):
        if ifrand == True:
            encoder_input, decoder_input, neA, S, fact_mat, fact_indices = random.choice(data[bucket_id])
        else:
            encoder_input, decoder_input, neA, S, fact_mat, fact_indices = data[bucket_id][i+idx*batch_size]
            
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
        batch_kg_indices.extend([[i,x] for x in fact_indices])

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
            batch_decoder_input[batch_idx] = dec_inp

        batch_decoder_inputs.append(batch_decoder_input)
        batch_weights.append(batch_weight)
        batch_masks.append(batch_mask)


    return batch_encoder_inputs, batch_decoder_inputs, \
            batch_targets, batch_weights, batch_masks, \
            seq_len, neAs, Ss, batch_facts, batch_kg_indices


def ifchange_read_info_data_with_buckets(data_path, info_path, kb_path, max_size=None, level=0):
    buckets = _buckets
    dataset = [[] for _ in buckets]
    with tf.gfile.GFile(data_path, mode='r') as data_file, \
         tf.gfile.GFile(info_path, mode='rb') as info_file, \
         tf.gfile.GFile(kb_path, mode='rb') as kb_file, \
         open(FLAGS.results_dir+'/test_kws.txt','r') as fkw, \
         open(FLAGS.results_dir+'/ifchange_kws.txt','w') as fif:

        info = pickle.load(info_file)
        kbs = pickle.load(kb_file)
        reduced_kbs = [kb for kb in kbs if len(kb) > 0]
        source = data_file.readline()
        target = data_file.readline()
        kws = fkw.readlines()

        if level == 1:#TODO
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
            fact_indices = []#
            new_pairs = []
            delete_pairs = []
            cur_s = []
            for j, triple in enumerate(fact):
                if len(triple) == 3:
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
                        nt = triple[2]

                    if level < 0:#TODO
                        vec = np.concatenate((data_utils.node_dict[data_utils.str_nodes[triple[0]]],
                                              data_utils.edge_type_dict[data_utils.edge_types[triple[1]]],
                                              data_utils.node_dict[data_utils.str_nodes[nt]]))
                        fact_mat.append(vec)
                        fact_indices.append(nt)
                if len(fact_mat) >= data_utils.triple_num and level < 0:
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
                for triple in fact:
                    vec = np.concatenate((data_utils.node_dict[data_utils.str_nodes[triple[0]]],
                                          data_utils.edge_type_dict[data_utils.edge_types[triple[1]]],
                                          data_utils.node_dict[data_utils.str_nodes[triple[2]]]))
                    fact_mat.append(vec)
                    fact_indices.append(triple[2])
                    if len(fact_mat) >= data_utils.triple_num:
                        break
            for _ in range(data_utils.triple_num - len(fact_mat)):
                fact_mat.append(np.zeros(data_utils.kbembed_size,))
                fact_indices.append(data_utils.kdim)


            stored = 0
            neA = fact
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    dataset[bucket_id].append([ source_ids, target_ids, neA, S, fact_mat, fact_indices ])
                    stored = 1
                    break
            if stored == 0:#truncate the length
                dataset[-1].append([ source_ids[:buckets[-1][0]], target_ids[:buckets[-1][1]], neA, S, fact_mat, fact_indices ])
                    
            # next loop
            source = data_file.readline()
            target = data_file.readline()

    return dataset, change_num
