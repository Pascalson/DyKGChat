import os
import time
import jieba
from subgraph import *

def norm(fchat, nodes, edges, spk_map):

    if not os.path.exists('kb_entities_dict.txt'):
        print('Do not find kb_entites_dict, create one ...')
        with open('kb_entities_dict.txt','w') as fentity:
            for n in nodes:
                word = ''.join(n.split())
                fentity.write(word+' 1000\n')
        print('Complete creating kb_entities_dict.')
    else:
        print('Already exists kb_entities_dict, loading ...')
    jieba.set_dictionary('kb_entities_dict.txt')
    jieba.initialize()

    raw_chats = fchat.readlines()
    chats, normalized_contexts, alligned_entities, alligned_dykb = {}, {}, {}, {}
    for _, line in enumerate(raw_chats):
        scene_no, speaker, scene, context = line.split('|')
        scene_no = scene_no.strip()
        speaker = speaker.strip()
        scene = scene.strip()
        context = context.strip()
        
        norm_spk = spk_map[speaker]
        norm_scene = jieba.lcut(scene, HMM=False)
        norm_kb_scene = [i for i in norm_scene if i in nodes]
        norm_context = jieba.lcut(context, HMM=False)
        norm_kb_context = [i for i in norm_context if i in nodes]
        if scene_no not in chats:
            chats[scene_no] = []
        chats[scene_no].append((norm_spk, norm_kb_scene, norm_kb_context, norm_context))

    count = 0
    subkb = SubGraph(nodes, edges)
    for scene_no, lines in chats.items():
        count += 1
        print('count {}: {}'.format(count, scene_no))
        start_time = time.time()
        content_time, kb_time = 0, 0
        normalized_contexts[scene_no] = []
        alligned_entities[scene_no] = []
        alligned_dykb[scene_no] = []
        for line1, line2 in zip(lines[:-1], lines[1:]):
            normalized_contexts[scene_no].append((' '.join(line1[3]), ' '.join(line2[3])))
            sources = list(set(line1[0] + line2[0] + line1[1] + line2[1] + line1[2]))
            alligned_entities[scene_no].append(sources)
            content_time += time.time()-start_time
            start_time = time.time()
            subkb.sample(sources, list(set(line2[2])))
            alligned_dykb[scene_no].append(subkb.triples)
            kb_time += time.time()-start_time
        print('avg content time:{}'.format(content_time/len(lines)))
        print('avg kb time:{}'.format(kb_time/len(lines)))

    return normalized_contexts, alligned_entities, alligned_dykb


def get_spk_map(fspk):
    spk_map = {}
    for _, line in enumerate(fspk):
        alias, _, norm_spks = line.split('|')
        alias = alias.strip()
        norm_spks = norm_spks.strip()
        if norm_spks != '':
            norm_spks = norm_spks.strip().split()
            spk_map[alias] = norm_spks
        else:
            spk_map[alias] = [alias]
    return spk_map
