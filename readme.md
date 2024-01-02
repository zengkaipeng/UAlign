## ENV

环境配置文件放在 chemistry.yml 里面，按照cuda服务器配置的，如果要用信创到时候找一下管理我把镜像给你

## Data

数据应该被放在同一个文件夹下，按照划分分为 `canonicalized_raw_train.csv`, `canonicalized_raw_val.csv`和 `canonicalized_raw_test.csv` 三个文件，我会把剩下的数据集整理好发私法

## Pretrain (First stage training)

pretrain 任务主要是简历graph和smiles之间的映射关系，使用的文件是 `pretrain.py` 推荐命令

`python pretrain.py --dim 512 --n_layer 8 --data_path <folder of dataset> --bs 128 --epoch 200 --early_stop 15 --device <device id> --lr 1.25e-4 --gnn_type gat --dropout 0.1 --base_log <folder to store log and model> --transformer --update_gate cat --token_path <token.json>  --aug_prob 1 --lrgamma 0.993 --warmup 4`

-  `token.json` 包含了所有分子经过tokenizer之后的token，会随着数据集更新
- 可以调整 `lr` 和 lr_scheduler 的 `lrgamma` ，现在这个配置可以用
- layer 和 dim 等模型架构参数不改动
- batch size 128 时大概需要20G显存，最好不要超过512



