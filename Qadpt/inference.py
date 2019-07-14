import sys
import info_data_utils as data_utils
from compute_knowledge import *
import tensorflow as tf
from main import get_batch_with_buckets
from main import _buckets
import math
import code
import copy
import collections
import queue
KG = data_utils.KG

def check_info(vec):
    tmp = vec.tolist()
    for ids, num in enumerate(tmp):
        if num == 1.:
            print(ids)

def sources(S):
    S_indices = np.nonzero(S)[0]
    S_arr = [data_utils.str_nodes[s_idx] for s_idx in S_indices]
    print('Sources: {}'.format(S_arr))

def find_T_path(Tproj, neA, S, target):
    #neA = KG.neA_M
    S_indices = np.nonzero(S)[0]
    # find ne -> nt
    nt = data_utils.str_nodes.index(target)
    ne_arr = np.take(neA, indices=nt, axis=1)
    ne_indices = np.nonzero(ne_arr)
    # find ns -> ne
    ne_dict = {}
    print(ne_indices)
    for ne in ne_indices[0]:
        print(ne)
        ns_idx = int(math.floor(ne/10))
        ns_name = data_utils.str_nodes[ns_idx]
        if ns_idx not in S_indices:
            find_T_path(Tproj, neA, S, ns_name)
        ne_prob = Tproj[ns_idx][ne] / np.sum(Tproj[ns_idx])
        et = int(ne - ns_idx*10)
        if et == 9:
            e_type = 'ToDumb'
        else:
            e_type = data_utils.edge_types[et]
        ne_dict[ns_name] = (e_type, ne, ne_prob)
    print('{} might be reached by: {}'.format(target, ne_dict))

def find_S_path(Tproj, neA, S, target):
    S_indices = np.nonzero(S)[0]
    probs = [1.0 for _ in S_indices]
    paths = [[s] for s in S_indices]
    print(paths)
    while True:
        new_S_indices = []
        new_paths = []
        new_probs = []
        count = 0
        for i, s in enumerate(S_indices):
            print(paths[i])
            if s == len(data_utils.str_nodes):
                count += 1
                new_S_indices.append(s)
                new_probs.append(probs[i])
                new_paths.append(paths[i])
                if len(new_S_indices) > 10:
                    new_S_indices.pop(new_probs.index(min(new_probs)))
                    new_paths.pop(new_probs.index(min(new_probs)))
                    new_probs.pop(new_probs.index(min(new_probs)))
                continue
            ne_indices = np.nonzero(Tproj[s])[0]
            for ne in ne_indices:
                ne_prob = Tproj[s][ne] / np.sum(Tproj[s])
                nt_indices = np.nonzero(neA[ne])[0]
                for nt in nt_indices:
                    new_S_indices.append(nt)
                    new_path = copy.deepcopy(paths[i] + [int(ne - s*10), nt])
                    new_paths.append(new_path)
                    #new_probs.append(probs[i] * ne_prob)
                    new_probs.append(ne_prob)
                    if len(new_S_indices) > 10:
                        new_S_indices.pop(new_probs.index(min(new_probs)))
                        new_paths.pop(new_probs.index(min(new_probs)))
                        new_probs.pop(new_probs.index(min(new_probs)))

        S_indices = copy.deepcopy(new_S_indices)
        paths = copy.deepcopy(new_paths)
        probs = copy.deepcopy(new_probs)
        print(count)
        if count == len(S_indices):
            break


    for path, prob in zip(paths, probs):
        print_path = []
        for i, v in enumerate(path):
            if i % 2 == 1:
                if v == 9:
                    print_path.append('ToEND')
                else:
                    print_path.append(data_utils.edge_types[v])
            else:
                if v == 174:
                    print_path.append('END')
                else:
                    print_path.append(data_utils.str_nodes[v])
        print('{}:{}'.format(print_path, prob))



def inference(sess, model, vocabs, data=None, batch_size=1, feed_prev=True):

    if feed_prev == False and data == None:
        raise ValueError("feed_prev cannot turn on as True when there is no access to data.")
    elif feed_prev != False and feed_prev != True:
        raise ValueError("feed_prev must be either True or False.")

    global KG

    data_set, data_buckets_scale = None, None
    if data != None:
        data_set, data_buckets_scale = data
    vocab, rev_vocab = vocabs

    if data_set == None or data_buckets_scale == None:
        sys.stdout.write('info> ')
        sys.stdout.flush()
        infos = sys.stdin.readline()

    sys.stdout.write('> ')
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    
    while sentence:

        if sentence.strip() == 'exit()':
            break

        if data_set != None and data_buckets_scale != None:
            random_number_01 = np.random.random_sample()
            print('random_number_01:{}'.format(random_number_01))
            code.interact(local=dict(globals(), **locals()))
            bucket_id = min([i for i in range(len(data_buckets_scale))
                             if data_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, targets, weights, masks, seq_lens, neAs, Ss = \
                get_batch_with_buckets(data_set, batch_size, bucket_id)

            if feed_prev == True:
                decoder_inputs = [[data_utils.GO_ID for _ in range(batch_size)]]
            elif feed_prev == False:
                decoder_inputs = decoder_inputs

        elif data_set != None:#TODO
            dialog_id = int(infos)
            dialog = data_set[dialog_id]
            batch_size = round(len(dialog) / 2)
            encoder_inputs = [ [] for i in range(_buckets[-1][0])]
            seq_lens = []
            neAs, Ss = [], []
            targets = []
            for idx in range(batch_size):
                enc_sentence, neA, S = dialog[2*idx]
                dec_sentence, _, _ = dialog[2*idx+1]
                token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(enc_sentence), vocab, normalize_digits=False)
                encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                seq_lens.append(len(token_ids))
                token_ids = list(reversed(token_ids)) + encoder_pad
                for t, token in enumerate(token_ids):
                    encoder_inputs[t].append(token)
                neAs.append(neA)
                Ss.append(S)
                targets.append(dec_sentence)
            decoder_inputs = [[data_utils.GO_ID for _ in range(batch_size)]]
                
        else:#TODO
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
            encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
            seq_lens = [len(token_ids)]
            token_ids = list(reversed(token_ids)) + encoder_pad
            encoder_inputs = []
            for idx in token_ids:
                encoder_inputs.append([idx])

            decoder_inputs = [[data_utils.GO_ID]]

            infos = infos.strip().split()
            info_vec = [get_seq_vec(infos, data_utils.str_nodes)]


        code.interact(local=dict(globals(), **locals()))
        #AKFVs = compute_Laplacian(info_vec, KG.M)
        #code.interact(local=dict(globals(), **locals()))
        #kinfos = get_nodes_info(data_utils.str_nodes, KG.get_edges(), data_utils.edge_types)
        #kvec = [np.reshape(np.concatenate((np.reshape(AKFV,(-1,1)), kinfos), axis=1) \
        #        , (len(data_utils.nodes), len(data_utils.edge_types)+1, 1)) \
        #        for AKFV in AKFVs]
        #code.interact(local=dict(globals(), **locals()))

        #try:
        if feed_prev == True:#TODO
            outputs, enc_state, a1s, kdists, Ndists, logits, Rdebugs \
                = model.dynamic_decode(sess, encoder_inputs, seq_lens, \
                                       decoder_inputs, neAs, Ss, -1)
        elif feed_prev == False:
            eval_loss, eval_decKB_loss, eval_decN_loss, eval_ptr_loss, \
                    outputs, a1s, kdists, Ndists, Rdebugs \
                = model.train_step(sess, encoder_inputs, \
                                   decoder_inputs, targets, weights, masks, \
                                   bucket_id, seq_lens, neAs, Ss, forward=True)

        for i in range(batch_size):
            
            subouts = [output_ids[i] for output_ids in outputs]
            if data_utils.EOS_ID in subouts:
                subouts = subouts[:subouts.index(data_utils.EOS_ID)]

            a1_list = [a1[i] for a1 in a1s]
            kdist_list = [kdist[i] for kdist in kdists]
            Ndist_list = [Ndist[i] for Ndist in Ndists]

            #print(kdist_list)
            #print(a1_list)
            print(" ".join([tf.compat.as_str(rev_vocab[out]) for out in subouts]))

            if data_set != None and data_buckets_scale != None:
                gts = [target_ids[i] for target_ids in targets]
                if data_utils.EOS_ID in gts:
                    gts = gts[:gts.index(data_utils.EOS_ID)]
                print(" ".join([tf.compat.as_str(rev_vocab[gt]) for gt in gts]))
            elif data_set != None:#TODO
                print(targets[i])
        
            code.interact(local=dict(globals(), **locals()))

        if data_set == None or data_buckets_scale == None:
            sys.stdout.write('info> ')
            sys.stdout.flush()
            infos = sys.stdin.readline()

        sys.stdout.write('> ')
        sys.stdout.flush()
        sentence = sys.stdin.readline()
