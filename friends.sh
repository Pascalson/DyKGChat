batchsize=32
size=128
numlayers=1
lr=0.5
lrdecay=0.99
config=""

vocabsize=20000
factsize=98
buckets='[(10,10),(15,15),(25,25),(30,30)]'

if [[ ("$method" == 'pred_acc') || ("$method" == 'eval_pred_acc') \
    || ("$method" == 'ifchange') ]]; then
    batchsize=125
else
    batchsize=$batchsize
fi

kgpath_len=6
