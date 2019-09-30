import pickle
import numpy as np

def get_dynamics_stat(fin, ged, only_ged, proportion):
    kbs = pickle.load(fin)
    for i in range(len(kbs)):
        for j in range(i, len(kbs)):
            deleted_edges = set(kbs[i]) - set(kbs[j])
            inserted_edges = set(kbs[j]) - set(kbs[i])
            ged.append(len(list(deleted_edges)) + len(list(inserted_edges)))
            if len(kbs[i]) != 0 and len(kbs[j]) != 0:
                only_ged.append(len(list(deleted_edges)) + len(list(inserted_edges)))
                proportion.append((len(list(deleted_edges)) + len(list(inserted_edges)))/len(kbs[i]))
                proportion.append((len(list(deleted_edges)) + len(list(inserted_edges)))/len(kbs[j]))
            #print('complete {}/{}-{}/{}'.format(i, len(kbs), j, len(kbs)-i))
        print('complete {}/{}'.format(i, len(kbs)))

def get_stat(ged, name):
    q1, med, q2 = np.percentile(ged, [25, 50, 75])
    low = min(ged)
    high = max(ged)
    mean = float(sum(ged))/len(ged)
    return {'min':low,
            'q25':q1,
            'med':med,
            'q75':q2,
            'max':high,
            'mean':mean,
            'name':name}

import json
import pickle

if __name__ == '__main__':
    ged = []
    only_ged = []
    proportion = []
    with open('train_hgzhz.sp5','rb') as ftrain_kb, \
        open('dev_hgzhz.sp5','rb') as fvalid_kb, \
        open('test_hgzhz.sp5','rb') as ftest_kb, \
        open('ged_data_only.bin','wb') as fonly, \
        open('ged_proportion.bin','wb') as fprop, \
        open('stat_dynamics.json','w') as fout:
        get_dynamics_stat(ftrain_kb, ged, only_ged, proportion)
        get_dynamics_stat(fvalid_kb, ged, only_ged, proportion)
        get_dynamics_stat(ftest_kb, ged, only_ged, proportion)
        
        pickle.dump(only_ged, fonly)
        pickle.dump(proportion, fprop)
        all_stat = []
        all_stat.append(get_stat(ged, 'all'))
        all_stat.append(get_stat(only_ged, 'only_with_kb'))
        all_stat.append(get_stat(proportion, 'only_with_kb_proportion'))
        json.dump(all_stat, fout, indent=2)
