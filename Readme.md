# DyKGChat
This project is the implementation of our paper **DyKgChat: A Multi-domain Chit-chat Dialogue Generation Dataset Grounding on Dynamic Knowledge Graphs**.


## Requirements
* jieba
* python3
* tensorflow r1.13

## Files
* `data/`: the collected data `hgzhz/` and `friends/` as well as their trained TransE
* `model_ckpts/`: the trained models
* `Qadpt/`: the programs


## Usage
* clone the repository and switch to directory `Qadpt/`
```
$cd Qadpt/
```

* testing hgzhz (the following commands must be in order)
```
$bash run.sh -1 pred_acc Qadpt
$bash run.sh -1 ifchange Qadpt
$bash run.sh -1 eval_pred_acc Qadpt
```

* testing friends
```
$bash frun.sh -1 pred_acc Qadpt
$bash frun.sh -1 ifchange Qadpt
$bash frun.sh -1 eval_pred_acc Qadpt
```

The automatic evaluation results will be printed on the screen, and some files will be outputed to `Qadpt/hgzhz_results/` or `Qadpt/friends_results/`.

The default `ifchange` evaluates **Last-1** score. To change to **random** or **Last-2**, modify the `line 464` in `main.py` to `level=-1` or `level=1`.


* training hgzhz
```
$bash run.sh 0 None Qadpt_new
```

* testing friends
```
$bash frun.sh 0 None Qadpt_new
```

The trained model will be stored in `model_ckpts/hgzhz/Qadpt_new/` or `model_ckpts/friends/Qadpt_new/`
