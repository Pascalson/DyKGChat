batchsize=32
size=256
numlayers=1
lr=0.5
lrdecay=0.99
config=""

vocabsize=4000
factsize=176
buckets='[(10,10),(20,20),(30,40),(45,55)]'

if [[ ("$method" == 'pred_acc') || ("$method" == 'eval_pred_acc') \
    || ("$method" == 'ifchange') ]]; then
    batchsize=125
else
    batchsize=$batchsize
fi

kgpath_len=6
