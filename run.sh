datadir=data
gpu=$1
method=$2
model=$3
data_name=$4
exp_name=$5

hops_num=1
change_level=1

if [ $# != 5 ]; then
  echo "usage: $0 <GPU_ID> <method> <model> <data> <exp_name>"
  echo "e.g.: $0 1 train Qadpt hgzhz Qadpt_hgzhz_v1_1"
  exit 1;
fi

resultsdir=results/$data_name/$exp_name
datapath=$datadir/$data_name/$data_name.txt
modeldir=model_ckpts/$data_name/$exp_name
datatype=test

. $data_name.sh

echo "$task; VOCAB SIZE:$vocabsize; SIZE:$size*$numlayers"
echo "BatchSize:$batchsize;"
echo "LearningRate:$lr; DecayFactor:$lrdecay"

CUDA_VISIBLE_DEVICES=$gpu python3 -i main.py \
  --model=$model \
  --model-dir=$modeldir \
  --results-dir=$resultsdir \
  --data-path=$datapath \
  --data-type=$datatype \
  --size=$size \
  --num-layers=$numlayers \
  --hops-num=$hops_num \
  --kgpath-len=$kgpath_len \
  --vocab-size=$vocabsize \
  --fact-size=$factsize \
  --lr=$lr \
  --lr-decay=$lrdecay \
  --buckets=$buckets \
  --batch-size=$batchsize \
  --test-type=$method \
  --change-level=$change_level
