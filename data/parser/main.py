import json
import pickle
import collections
import os

import read_kb
import normalize
import split

data_name = 'hgzhz'

if __name__ == '__main__':

    with open('raw_data/chats_n_scenes.txt','r') as fchat, \
        open('raw_data/people_mapping.txt','r') as fspk, \
        open('train_'+data_name+'.txt','w') as ftrain, \
        open('train_'+data_name+'.info','wb') as ftrain_info, \
        open('train_'+data_name+'.sp5','wb') as ftrain_kb, \
        open('dev_'+data_name+'.txt','w') as fvalid, \
        open('dev_'+data_name+'.info','wb') as fvalid_info, \
        open('dev_'+data_name+'.sp5','wb') as fvalid_kb, \
        open('test_'+data_name+'.txt','w') as ftest, \
        open('test_'+data_name+'.info','wb') as ftest_info, \
        open('test_'+data_name+'.sp5','wb') as ftest_kb:

        nodes, edges = read_kb.read_in_graph('.')
        if not os.path.exists('temp_norm.dict'):
            spk_map = normalize.get_spk_map(fspk)
            print(spk_map)
            normalized_contexts, alligned_entities, alligned_dykb = normalize.norm(fchat, nodes, edges, spk_map)
            pickle.dump((normalized_contexts, alligned_entities, alligned_dykb), open('temp_norm.dict','wb'))
        else:
            (normalized_contexts, alligned_entities, alligned_dykb) = pickle.load(open('temp_norm.dict','rb'))
        entities_occurs = split.shuffle( nodes, list(sorted(edges.keys())), \
            normalized_contexts, alligned_entities, alligned_dykb, \
            [ftrain, fvalid, ftest], \
            [ftrain_info, fvalid_info, ftest_info], \
            [ftrain_kb, fvalid_kb, ftest_kb], \
            [0.85,0.05,0.1])

    with open('for_kb_cloud.txt','w') as fkb:
        num_kb_appears = 0
        kb_counts = {}
        for n in nodes:
            box = []
            for i in [0, 1, 2]:
                if n in entities_occurs[i]:
                    box.append(entities_occurs[i][n])
                    num_kb_appears += entities_occurs[i][n]
                    for _ in range(entities_occurs[i][n]):
                        fkb.write(n)
                        fkb.write(' ')
                    fkb.write('\n')
                else:
                    box.append(0)
            print('{}:{}'.format(n, box))
            kb_counts[n] = box
        print('# kb entities appears:{}'.format(num_kb_appears))
