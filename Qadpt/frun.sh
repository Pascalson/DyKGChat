datadir=friends_data
gpu=$1
testtype=$2
mdlname=$3

batchsize=32
size=128
numlayers=1
lr=0.5
lrdecay=0.99
config=""

resultsdir=friends_result
datapath=$datadir/friends.txt
modeldir=friends_exp/$mdlname
vocabsize=20000
factsize=98
buckets='[(10,10),(15,15),(25,25),(30,30)]'

if [[ ("$testtype" == 'data_argmax') || ("$testtype" == 'check_Qadpt') ]]; then
    batchsize=3
elif [[ ("$testtype" == 'pred_acc') || ("$testtype" == 'eval_pred_acc') \
    || ("$testtype" == 'ifchange') ]]; then
    batchsize=125
elif [[ ("$testype" == 'eval') ]]; then
    batchsize=5
else
    batchsize=$batchsize
fi


echo "$task; VOCAB SIZE:$vocabsize; SIZE:$size*$numlayers"
echo "BatchSize:$batchsize;"
echo "LearningRate:$lr; DecayFactor:$lrdecay"

CUDA_VISIBLE_DEVICES=$gpu python3 -i main.py \
  --lr=$lr \
  --lr-decay=$lrdecay \
  --model-dir=$modeldir \
  --data-path=$datapath \
  --size=$size \
  --num-layers=$numlayers \
  --vocab-size=$vocabsize \
  --fact-size=$factsize \
  --buckets=$buckets \
  --batch-size=$batchsize \
  --test-type=$testtype \
  --results-dir=$resultsdir \
  $config
