# UAlign

Official implementation for paper:

UAlign: Pushing the Limit of Template-free Retrosynthesis Prediction with Unsupervised SMILES Alignment

## Environment

two anaconda environments are provided, corresponding to CUDA 10.2 and CUDA 11.3 respectively. Use the following commands to create the environment for running our code.

```shell
conda env create -f env_cu102.yml # for CUDA 10.2
conda env create -f env_cu113.yml # for CUDA 11.3
```

## Data and Checkpoints

The raw data, processed data, checkpoints and the predicted results can be accessed via [link](https://drive.google.com/drive/folders/1hADgJ_Sga7xVao73ChlQy74bieS-5kYX?usp=drive_link). The directory structure should be as follows:

```
UAlign
├───checkpoints
│   ├───USPTO-50K
│   │       ├───class_unknown.pkl
│   │       ├───class_unknown.pth
│   │       ├───class_known.pkl
│   │       └───class_known.pth
│   │       
│   ├───USPTO-FULL
│   │       ├───model.pth
│   │       └───token.pkl
│   │       
│   └───USPTO-MIT
│           ├───model.pth
│           └───token.pkl
│           
├───Data
|   ├───USPTO-50K
|   │       ├───canonicalized_raw_test.csv
|   │       ├───canonicalized_raw_val.csv
|   │       ├───canonicalized_raw_train.csv
|   │       ├───raw_test.csv
|   │       ├───raw_val.csv
|   │       └───raw_train.csv
|   │       
|   ├───USPTO-MIT
|   │       ├───canonicalized_raw_train.csv
|   │       ├───canonicalized_raw_val.csv
|   │       ├───canonicalized_raw_test.csv
|   │       ├───valid.txt
|   │       ├───test.txt
|   │       └───train.txt
|   │       
|   └───USPTO-FULL
|           ├───canonicalized_raw_val.csv
|           ├───canonicalized_raw_test.csv
|           ├───canonicalized_raw_train.csv
|           ├───raw_val.csv
|           ├───raw_test.csv
|           └───raw_train.csv
|                     
└───predicted_results
    ├───USPTO-50K
    │       ├───answer-1711345166.9484136.json
    │       └───answer-1711345359.2533984.json
    │       
    ├───USPTO-MIT
    │       ├───10000-20000.json
    │       ├───30000-38648.json
    │       ├───20000-30000.json
    │       └───0-10000.json
    │       
    └───USPTO-FULL
            ├───75000-96014.json
            ├───25000-50000.json
            ├───50000-75000.json
            └───0-25000.json
```

- Data
    - The raw data of the USPTO-50K dataset and USPTO-FULL dataset is stored in the corresponding folders in the files $\texttt{raw\_train.csv}$, $\texttt{raw\_val.csv}$, and $\texttt{raw\_test.csv}$. The raw data of USPTO-MIT dataset are named $\texttt{train.txt}$, $\texttt{valid.txt}$ and $\texttt{test.txt}$ under the folder $\texttt{USPTO-MIT}$. 
    - All the processed data are named $\texttt{canonicalized\_raw\_train.csv}$ , $\texttt{canonicalized\_raw\_val.csv}$ and $\texttt{canonicalized\_raw\_test.csv}$ and are put in the corresponding folders respectively. **If you want to use your own data for training, please make sure the your files have the same format and the same name as the processed ones.**
- Checkpoints
    - Every checkpoint needs to be used together with its corresponding tokenizer. The tokenizers are stored as $\texttt{pkl}$ files, while the trained model weights are stored in $\texttt{pth}$ files. The matching model weights and tokenizer have the same name and are placed in the same folder.
- predicted_results
    - In the $\texttt{USPTO-FULL}$ and $\texttt{USPTO-MIT}$ folders, there is only one set of experimental results in each. They are divided into different files based on the index of the data. 
    - In USPTO-50K, there are two sets of experimental results. The file $\texttt{answer-1711345166.9484136.json}$ corresponds to the setting of reaction class unknowns, while $\texttt{answer-1711345359.2533984.json}$ corresponds to the setting of reaction class known. 
    - Each json file contains raw data for testing, the model's prediction results, corresponding logits, and also includes the checkpoints information used to generate this json file.



## Data Preprocess

The data should have 

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



