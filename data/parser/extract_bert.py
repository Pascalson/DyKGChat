import numpy as np
from bert_serving.client import BertClient
bc = BertClient()

with open(file_name+'.txt','r') as fin, \
    open(file_name+'.bert','wb') as fout:
