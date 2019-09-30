import random
import pickle
import collections

def shuffle(nodes, index_edges, normalized_contexts, alligned_entities, alligned_dykb, fouts, finfos, fkbs, rates):
    keys = list(sorted(normalized_contexts.keys()))
    random.seed(10)
    random.shuffle(keys)
    train_boundary = round(rates[0]*len(keys))
    test_boundary = round((rates[0]+rates[1])*len(keys))
    print('train # dialogues:{}'.format(train_boundary))
    print('valid # dialogues:{}'.format(test_boundary-train_boundary))
    print('test # dialogues:{}'.format(len(keys)-test_boundary))
    print(len(keys))

    kb_counts = {n:0 for n in nodes}
    total_kb_count = 0
    num_turns_with_kb = 0
    num_dialogs_with_kb = 0
    total_tokens = {0:{}, 1:{}, 2:{}}
    entities_occurs = {0:{}, 1:{}, 2:{}}
    box_ali_entities = [[],[],[]]
    box_ali_dykb = [[],[],[]]
    max_size_dykb = 0
    for scene_no, lines in normalized_contexts.items():
        boundary_id = keys.index(scene_no)
        if boundary_id < train_boundary:
            box = 0#'train'
        elif boundary_id >= test_boundary:
            box = 2#'test'
        else:
            box = 1#'valid'

        flag_num_turns_with_kb = num_turns_with_kb
        for line, entities, dykb in zip(lines, alligned_entities[scene_no], alligned_dykb[scene_no]):
            for e in entities:
                if e not in entities_occurs[box]:
                    entities_occurs[box][e] = 1
                else:
                    entities_occurs[box][e] += 1
            if len(dykb) > 0:
                num_turns_with_kb += 1
            if len(dykb) > max_size_dykb:
                max_size_dykb = len(dykb)
            fouts[box].write(line[0]+'\n'+line[1]+'\n')
            for n in line[0].split():
                if n not in total_tokens[box]:
                    total_tokens[box][n] = 1
                else:
                    total_tokens[box][n] += 1
            if box == 2:
                for n in line[1].split():
                    if n in nodes:
                        kb_counts[n] += 1
                        total_kb_count += 1

        if num_turns_with_kb != flag_num_turns_with_kb:
            num_dialogs_with_kb += 1
        for n in line[1].split():
            if n not in total_tokens[box]:
                total_tokens[box][n] = 1
            else:
                total_tokens[box][n] += 1

        index_infos = [[nodes.index(i) for i in entities if i in nodes] for entities in alligned_entities[scene_no]]
        box_ali_entities[box].extend(index_infos)
        index_dykb = [[(nodes.index(i[0]), index_edges.index(i[1]), nodes.index(i[2])) for i in dykb] for dykb in alligned_dykb[scene_no]]
        box_ali_dykb[box].extend(index_dykb)
    for box in [0,1,2]:
        pickle.dump(box_ali_entities[box], finfos[box])
        pickle.dump(box_ali_dykb[box], fkbs[box])
        #print('{} | {} | {}'.format(line, entities, dykb))

    print('# dialogs with kb:{}'.format(num_dialogs_with_kb))
    print('# turns with kb:{}'.format(num_turns_with_kb))
    unique_tokens = []
    total_num_tokens = 0
    for box in [0,1,2]:
        unique_tokens += total_tokens[box].keys()
        total_num_tokens += sum(total_tokens[box].values())
    print('# unique tokens:{}'.format(len(list(set(unique_tokens)))))
    print('total # tokens:{}'.format(total_num_tokens))
    print('maximum # triples:{}'.format(max_size_dykb))
    kb_proportion = {}
    for key, value in kb_counts.items():
        kb_proportion[key] = value / total_kb_count
    print(collections.OrderedDict(sorted(kb_proportion.items(), key=lambda x: x[1])))

    return entities_occurs
