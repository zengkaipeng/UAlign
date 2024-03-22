## This REPO is for github

## ENV

环境配置文件放在 chemistry.yml 里面，按照cuda服务器配置的，如果要用信创到时候找一下管理我把镜像给你

## Data

数据应该被放在同一个文件夹下，按照划分分为 `canonicalized_raw_train.csv`, `canonicalized_raw_val.csv`和 `canonicalized_raw_test.csv` 三个文件，我会把剩下的数据集整理好发私法

## Pretrain (First stage training)

pretrain 任务主要是简历graph和smiles之间的映射关系，使用的文件是 `pretrain.py` 推荐命令

`python pretrain.py --dim 512 --n_layer 8 --data_path <folder of dataset> --bs 128 --epoch 200 --early_stop 15 --device <device id> --lr 1.25e-4 --gnn_type gat --dropout 0.1 --base_log <folder to store log and model> --transformer --heads 8 --update_gate cat --token_path <token.json>  --aug_prob 1 --lrgamma 0.993 --warmup 4`

-  `token.json` 包含了所有分子经过tokenizer之后的token，会随着数据集更新
- 可以调整 `lr` 和 lr_scheduler 的 `lrgamma` ，现在这个配置可以用
- layer 和 dim 等模型架构参数不改动
- batch size 128 时大概需要20G显存，最好不要超过512

## Stage II training

使用的文件是 `train_aug.py`

`python train_aug.py --dim 512 --n_layer 8 --data_path <fodler of dataset> --aug_prob 0.5 --heads 8 --warmup 4 --gnn_type gat --update_gat cat --transformer --gamma 0.993 --dropout 0.1 --data_path <folder of dataset> --bs 128 --epoch 300 --early_stop 15 --device <device id> --lr 1.25e-4 --base_log <folder containing logs and model> --accu 1 --step_start 20 --checkpoint <pretrained pth> --token_ckpt <pretrained tokenizer.pkl> --label_smoothing 0.1 (--use_class)`

- 不需要`token.json` 了，tokenizer要和pretrain的时候对齐
- 可以调整 `lr` 和 lr_scheduler 的 `gamma` ，`dropout`，`label_smoothing` 来找到最好的ckpt
- layer 和 dim 等模型架构参数不改动
- batch size 128 时大概需要20G显存
- step_start 是 lr_scheduler 开始做lr衰减的epoch， warmup是warmup的epoch数量， early_stop是检验模型是否停止训练的early stop的轮数
- --use_class 加与不加代表reaction known 和 unknown两种setting的实验，都要做

## find the best checkpoint

`python get_time_config.py --dir <folder containing logs and model> --topk <topk> --filter <str>`

filter 跟着的是一个 代表`dict`的`str`，可以用`python` 的 `eval()` 函数直接解析出来的，把想要控制的参数放在里面，比如我需要reaction class known的结果，那就 `--filter "{'use_class': True}"`，这个会列出greedy search evaluation 最好的`k`个 checkpoint和他们对应的信息



