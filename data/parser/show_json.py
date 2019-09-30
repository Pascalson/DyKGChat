#encoding=utf8
from read_kb import *
import json
import pickle

nodes, edges = read_in_graph('.')

with open('test_hgzhz.txt','r') as fin, \
    open('test_hgzhz.sp5','rb') as fkg, \
    open('test_hgzhz.info','rb') as fs, \
    open('test.json','w') as fout:
    
    kgs = pickle.load(fkg)
    ss = pickle.load(fs)
    
    all_data = []

    lines = fin.readlines()
    inputs = [lines[2*i] for i in range(int(len(lines)/2))]
    references = [lines[2*i+1] for i in range(int(len(lines)/2))]

    for idx, (inp, ref) in enumerate(zip(inputs, references)):
        all_data.append({
            'input':inp,
            'response':ref,
            'subkg':kgs[idx],
            'sources':ss[idx],
            's_words':[nodes[s] for s in ss[idx]]
        })
    print(all_data)
    json.dump(all_data, fout, indent=2, ensure_ascii=False)
