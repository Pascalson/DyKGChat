import numpy as np

def beam_search(model, encoder_inputs, encoder_len):
    # PREPARE PROB DISTRIBUTION
    #example_idx = 0#FIXME
    batch_size = len(encoder_inputs)
    beam_size = 5
    beam_encoder_inputs = []
    for l in range(_buckets[-1][0]):
        beam_encoder_inputs.append(
            np.array([encoder_inputs[l][example_idx] for example_idx in range(batch_size) \
                for _ in range(beam_size)], np.int32))
    beam_encoder_len = [encoder_len[example_idx] for example_idx in range(batch_size) \
        for _ in range(beam_size)]
    decoder_inputs = [[data_utils.GO_ID for _ in range(batch_size)]] \
        + [[data_utils.PAD_ID for _ in range(batch_size)] for _ in range(_buckets[-1][1]-1)]


    # START, TIME STEP 0
    time_step = 1
    outs = model.stepwise_test_beam(sess, encoder_inputs, encoder_len, decoder_inputs)#beam_sizexdecoder_size
    outs = outs[time_step-1]#FIXME
    top_k_tokens_ids = [np.argsort(outs[example_idx])[-beam_size:] for example_idx in range(batch_size)]
    paths = [[data_utils.GO_ID for _ in range(beam_size*batch_size)], \
            [top_token for example_idx in range(batch_size) for top_token in top_k_tokens_idx[example_idx]]]
    ended_paths = []
    probs = np.zeros((beam_size*batch_size))
    for example_idx in range(batch_size):
        for k in range(beam_size):
            probs[example_idx*beam_size+k] += np.log(outs[example_idx][top_k_tokens_ids[k]])
            ended_paths.append(data_utils.EOS_ID == paths[-1][example_idx*beam_size+k])
    paths += [[data_utils.PAD_ID for _ in range(beam_size*batch_size)] for _ in range(_buckets[-1][1]-time_step-1)]

    # START, TIME STEP 1-T
    for _ in range(_buckets[-1][1]-1):
        time_step += 1
        outs = model.stepwise_test_beam(sess, beam_encoder_inputs, beam_encoder_len, paths)
        outs = outs[time_step-1]#FIXME
        top_k_tokens_ids = [np.argsort(outs[j])[-beam_size:] for j in range(beam_size*batch_size)]
        candidates_number = [[] for _ in range(batch_size)]# FOR each encoder-inputs
        tmp_probs = [[] for _ in range(batch_size)]
        top_k_paths_ids = []
        candidates_scales = []
        for i in range(batch_size):
            for j in range(beam_size):
                if ended_paths[i*beam_size+j]:# IF previous decoded path is already ended
                    tmp_probs[i].extend([probs[i*beam_size+j]])
                    candidates_number[i].append(1)
                else:
                    tmp_probs[i].extend([probs[i*beam_size+j]+np.log(outs[i*beam_size+j][k]) \
                        for k in top_k_tokens_ids[i*beam_size+j]])
                    candidates_number[i].append(beam_size)
            top_k_paths_ids.append(np.argsort(tmp_probs[i])[-beam_size:])#SELECT the top $beamsize path from beam_size*beamsize for EACH encoder-inputs
            candidates_scales.append([sum(candidates_number[i][:j+1]) for j in range(beam_size)])
        # STORE PAST PATH
        tmp_paths = []
        for l in range(time_step):
            step = []
            for i in range(batch_size):
                for k in top_k_paths_ids[i]:
                    selected_old_path_id = min([j for j in range(beam_size) if candidates_scales[i][j] > k])#FIXME
                    step.append(paths[l][i*beam_size+selected_old_path_id])
            tmp_paths.append(step)
        paths = tmp_paths
        # APPEND NEW STEP
        step = []
        for i in range(batch_size):
            for j, k in enumerate(top_k_paths_ids[i]):
                expand_candidates_scales = [0] + candidates_scales[i]
                old_path_id = min([j for j in range(beam_size) if candidates_scale[i][j] > k])
                token_id = k - expand_candidates_scales[old_path_id]
                step.append(top_k_tokens_ids[old_path_id*batch_size+token_id])
                probs[i*beam_size+j] = tmp_probs[i][k]
        paths.append(step)
        # APPEND PADDING
        for i in range(batch_size):
            for j, k in enumerate(top_k_paths_ids[i]):
                ended_paths[i*beam_size+j] = False
                for l in range(time_step+1):
                    if paths[l][i*beam_size+j] == data_utils.EOS_ID:
                        ended_paths[i*beam_size+j] = True
        paths += [[data_utils.PAD_ID for _ in range(beam_size*batch_size)] for _ in range(_buckets[-1][1]-time_step-1)]
    return paths, probs

def MMI(decoder_inputs):
    lm_probs = model.lm_prob(sess, decoder_inputs)
    lm_probs = lm_probs[0]
    lm_prob = []
    lens = []
    for p in range(len(decoder_inputs[0])):
        tmp_prob = 0.0
        for l in range(len(decoder_inputs)-1):
            if l < 3:
                tmp_prob += np.log(lm_probs[l][p][decoder_inputs[l+1][p]])
            if decoder_inputs[l+1][p] == data_utils.EOS_ID:
                lens.append(l)
                break
        lm_prob.append(tmp_prob)
        if len(lens) < len(lm_prob):
            lens.append(len(decoder_inputs)-1)
    return lm_prob, lens
