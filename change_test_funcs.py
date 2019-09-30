import tensorflow as tf

import data_utils
import args
FLAGS = args.FLAGS


KG = data_utils.KG
edges = KG.get_edges()
edim = len(data_utils.edge_types)

def ifchange(sess, results_dirs, model, vocabs, data, batch_size=1000, feed_prev=True):

    if feed_prev != False and feed_prev != True:
        raise ValueError("feed_prev must be either True or False.")
    vocab, rev_vocab = vocabs
    encoder_inputs, decoder_inputs, targets, weights, \
        masks, seq_lens, neAs, Ss, facts, kg_indices = data

    with open(results_dirs[0],'r') as fin:
        cur_idx = len(fin.readlines())

    change_num = 0
    # RUN
    if feed_prev == False:
        decoder_inputs = decoder_inputs
    elif feed_prev == True:
        decoder_inputs = [[data_utils.GO_ID for _ in range(batch_size)]]


    if feed_prev == False:
        eval_loss, outputs, soft_outs, a1s, kdists, Ndists, Rdebugs \
            = model.train_step(sess, encoder_inputs, \
                decoder_inputs, targets, \
                weights, masks, \
                -1, seq_lens, neAs, Ss, \
                facts, kg_indices, forward=True)
    elif feed_prev == True:
        outputs, enc_state, a1s, kdists, Ndists, logits, Rdebugs \
            = model.dynamic_decode(sess, \
                encoder_inputs, seq_lens, \
                decoder_inputs, neAs, Ss, facts, kg_indices, -1)

    with open(results_dirs[0],'a') as fout, \
        open(FLAGS.results_dir+'/test_outs.txt','r') as fin, \
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

            if strout != ori:
                have_changed += 1
            out_kws = [x for x in strout.split() if x in data_utils.str_nodes]
            for x in if_kws[i+cur_idx].strip().split():
                if x in out_kws:
                    accu_changed += 1
                    break

        return [have_changed, accu_changed]
