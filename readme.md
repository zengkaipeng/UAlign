# UAlign

Official implementation for paper:

[UAlign: Pushing the Limit of Template-free Retrosynthesis Prediction with Unsupervised SMILES Alignment](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00877-2)

## Updates

- Refactored the repository layout: reusable model code is under `models/`, training/data utilities are under `utils/`, and decoder cache helpers are under `models/decoder/`.
- Added model architecture presets under `model_arch/` and standardized bundled token JSON files under `smiles_tokens/`.
- Merged batched inference and evaluation CLIs, with support for SMILES augmentation, reranking, directory evaluation, and cached decoding.
- Added KV-cache support for PyTorch 1.x inference.
- Unified Stage I / Stage II training helpers, added deterministic `--log_name` outputs, and added `--scilence` for non-interactive launcher runs.
- Added training-time unknown-token checks and Stage I augmentation guards for invalid or very large randomized molecule pairs.

## Environment

two anaconda environments are provided, corresponding to CUDA 10.2 and CUDA 11.3 respectively. Use the following commands to create the environment for running our code.

```shell
conda env create -f env_config/env_cu102.yml # for CUDA 10.2
conda env create -f env_config/env_cu113.yml # for CUDA 11.3
```

## Data and Checkpoints

The raw data, processed data, checkpoints and the predicted results can be accessed via [link](https://drive.google.com/drive/folders/1hADgJ_Sga7xVao73ChlQy74bieS-5kYX?usp=drive_link). The directory structure should be as follows:

```text
UAlign/
|- checkpoints/
|  |- USPTO-50K/
|  |  |- class_unknown.pkl
|  |  |- class_unknown.pth
|  |  |- class_known.pkl
|  |  `- class_known.pth
|  |- USPTO-FULL/
|  |  |- model.pth
|  |  `- token.pkl
|  `- USPTO-MIT/
|     |- model.pth
|     `- token.pkl
|- Data/
|  |- USPTO-50K/
|  |  |- canonicalized_raw_test.csv
|  |  |- canonicalized_raw_val.csv
|  |  |- canonicalized_raw_train.csv
|  |  |- raw_test.csv
|  |  |- raw_val.csv
|  |  `- raw_train.csv
|  |- USPTO-MIT/
|  |  |- canonicalized_raw_train.csv
|  |  |- canonicalized_raw_val.csv
|  |  |- canonicalized_raw_test.csv
|  |  |- valid.txt
|  |  |- test.txt
|  |  `- train.txt
|  `- USPTO-FULL/
|     |- canonicalized_raw_val.csv
|     |- canonicalized_raw_test.csv
|     |- canonicalized_raw_train.csv
|     |- raw_val.csv
|     |- raw_test.csv
|     `- raw_train.csv
`- predicted_results/
   |- USPTO-50K/
   |  |- answer-1711345166.9484136.json
   |  `- answer-1711345359.2533984.json
   |- USPTO-MIT/
   |  |- 10000-20000.json
   |  |- 30000-38648.json
   |  |- 20000-30000.json
   |  `- 0-10000.json
   `- USPTO-FULL/
      |- 75000-96014.json
      |- 25000-50000.json
      |- 50000-75000.json
      `- 0-25000.json
```

- Data
    - The raw data of the USPTO-50K dataset and USPTO-FULL dataset is stored in the corresponding folders in the files `raw_train.csv`, `raw_val.csv`, and `raw_test.csv`. The raw data of USPTO-MIT dataset are named `train.txt`, `valid.txt` and `test.txt` under the folder `USPTO-MIT`. 
    - All the processed data are named `canonicalized_raw_train.csv` , `canonicalized_raw_val.csv` and `canonicalized_raw_test.csv` and are put in the corresponding folders respectively. **If you want to use your own data for training, please make sure the your files have the same format and the same name as the processed ones.**
    
- Checkpoints
    - Every checkpoint needs to be used together with its corresponding tokenizer. The tokenizers are stored as `pkl` files, while the trained model weights are stored in `pth` files. The matching model weights and tokenizer have the same name and are placed in the same folder.
    
    - The parameters of checkpoint for USPTO-50K are
    
      ```
      dim: 512
      n_layer: 8
      heads: 8
      negative_slope: 0.2
      ```
    
      The parameters of checkpoint for USPTO-MIT and USPTO-FULL are
      ```
      dim: 768
      n_layer: 8
      heads: 12
      negative_slope: 0.2
      ```
      Matching model architecture presets are provided under `model_arch/`:
      `uspto_50k.json`, `uspto_mit.json`, and `uspto_full.json`.
      
    
- predicted_results
    - In the `USPTO-FULL` and `USPTO-MIT` folders, there is only one set of experimental results in each. They are divided into different files based on the index of the data. 
    - In USPTO-50K, there are two sets of experimental results. The file `answer-1711345166.9484136.json` corresponds to the setting of reaction class unknown, while `answer-1711345359.2533984.json` corresponds to the setting of reaction class known. 
    - Each `json` file contains raw data for testing, the model's prediction results, corresponding logits, and also includes the checkpoints information used to generate this `json` file.

## Data Preprocess

We provide the data preprocess scripts in folder `data_proprocess`. The atom-mapping numbers of each reaction are reassigned according to the canonical ranks of atoms of the product to avoid information leakage. USPTO-50K and USPTO-FULL share the same CSV canonicalization implementation, but they must use different class modes.

**Important class-mode setting:**

- USPTO-50K uses `preserve`: keep the reaction class from the raw CSV.
- USPTO-FULL uses `minus_one`: write `class = -1` for every reaction.
- USPTO-MIT uses its own text-file preprocessing script and also writes `class = -1`.

Using the wrong class mode will produce processed data with incorrect class labels.

```shell
# USPTO-50K, multiprocessing enabled by default with --num_procs 4
python data_proprocess/canonicalize_data_50K.py --filename $path_of_raw_csv
python data_proprocess/canonicalize_data_50K.py --filename $path_of_raw_csv --num_procs 8

# USPTO-FULL, multiprocessing enabled by default with --num_procs 4
python data_proprocess/canonicalize_data_full.py --filename $path_of_raw_csv
python data_proprocess/canonicalize_data_full.py --filename $path_of_raw_csv --num_procs 8
```

The compatibility wrappers above set the correct class mode automatically and do not expose `--class_mode`. If you call the shared CSV implementation directly, specify the class mode explicitly:

```shell
python data_proprocess/canonicalize_data_csv.py --filename $path_of_50k_raw_csv --class_mode preserve
python data_proprocess/canonicalize_data_csv.py --filename $path_of_full_raw_csv --class_mode minus_one
```

The script for USPTO-MIT processes all the files together, which can be used by

```shell
python data_proprocess/canonicalize_data_mit.py --dir $folder_of_raw_data --output_dir $output_dir
```

The `$folder_of_raw_data` should contain the following files: `train.txt`, `valid.txt`, and `test.txt`. The USPTO-MIT script writes `canonicalized_raw_train.csv`, `canonicalized_raw_val.csv`, and `canonicalized_raw_test.csv` to `$output_dir`.

**For the detail about data preprocess, please refer to the article.**

## Generating Tokens

To build the tokenizer, we need a list of all shown tokens. You can use the following command to generate the token list and store it in a JSON file.

```shell
python generate_tokens.py $file_1 $file_2 ... $file_n $token_list.json
```

The script can accept multiple files as input and the last position should be the path of file to store the token list. The files should have the same format as the processed dataset.

## Stage I training

For reproducing the release-style experiments, prefer the scripts under `launchers/`. The command templates below are mainly for custom training runs.

Use the following command for training the first stage:

```shell
python pretrain.py --model_arch_path model_arch/uspto_50k.json \
                   --data_path $folder_of_dataset \
                   --seed $random_seed \
                   --bs $batch_size \
                   --epoch $epoch_for_training \
                   --early_stop $epoch_num_for_checking_early_stop \
                   --device $device_id \
                    --lr $learning_rate \
                   --base_log $folder_for_logging \
                   --token_path $path_of_token_list \
                   --checkpoint $path_of_checkpoint \
                   --token_ckpt $path_of_checkpoint_for_tokenizer \
                   --lrgamma $decay_rate_for_lr_scheduler \
                   --warmup $epoch_num_for_warmup \
                   --accu $batch_num_for_gradient_accumulation \
                   --num_worker $num_worker_for_data_loader
```

If the checkpoints for model and tokenizer are provided, the path for token list is not necessary and will be ignored if you pass it to the arguments of the script. Also for data distributed training, you can use:

```shell
python ddp_pretrain.py --model_arch_path model_arch/uspto_50k.json \
                       --data_path $folder_of_dataset \
                       --seed $random_seed \
                       --bs $batch_size \
                       --epoch $epoch_for_training \
                       --early_stop $epoch_num_for_checking_early_stop \
                       --lr $learning_rate \
                       --base_log $folder_for_logging \
                       --token_path $path_of_token_list \
                       --checkpoint $path_of_checkpoint \
                       --token_ckpt $path_of_checkpoint_for_tokenizer \
                       --lrgamma $decay_rate_for_lr_scheduler \
                       --warmup $epoch_num_for_warmup \
                       --accu $batch_num_for_gradient_accumulation \
                       --num_worker $num_worker_for_data_loader \
                       --num_gpus $num_of_gpus_for_training \
                       --port $port_for_ddp_training
```

## Stage II training

Use the following command to train the second stage:

```shell
python train_trans.py --model_arch_path model_arch/uspto_50k.json \
                          --aug_prob $probability_for_data_augumentation \
                          --data_path $folder_of_dataset \
                          --seed $random_seed \
                          --bs $batch_size \
                          --epoch $epoch_for_training \
                          --early_stop $epoch_num_for_checking_early_stop \
                          --lr $learning_rate \
                          --base_log $folder_for_logging \
                          --token_path $path_of_token_list \
                          --checkpoint $path_of_checkpoint \
                          --token_ckpt $path_of_checkpoint_for_tokenizer \
                          --gamma $decay_rate_for_lr_scheduler \
                          --step_start $the_epoch_to_start_lr_decay \
                          --warmup $epoch_num_for_warmup \
                          --accu $batch_num_for_gradient_accumulation \
                          --num_worker $num_worker_for_data_loader \
                          --label_smoothing $label_smoothing_for_training \
                          [--use_class] #add it into command for reaction class known setting
```

If you want to train from scratch, pass the path of token list to the script and don't provide any checkpoints for it.  Also for data distributed training, you can use:

```shell
python ddp_train_trans.py --model_arch_path model_arch/uspto_50k.json \
				      --aug_prob $probability_for_data_augumentation \
				      --data_path $folder_of_dataset \
                      --seed $random_seed \
                      --bs $batch_size \
                      --epoch $epoch_for_training \
                      --early_stop $epoch_num_for_checking_early_stop \
                      --lr $learning_rate \
                      --base_log $folder_for_logging \
                      --token_path $path_of_token_list \
                      --checkpoint $path_of_checkpoint \
                      --token_ckpt $path_of_checkpoint_for_tokenizer \
                      --gamma $decay_rate_for_lr_scheduler \
                      --step_start $the_epoch_to_start_lr_decay \
                      --warmup $epoch_num_for_warmup \
                      --accu $batch_num_for_gradient_accumulation \
                      --num_worker $num_worker_for_data_loader \
                      --label_smoothing $label_smoothing_for_training \
                      --num_gpus $num_of_gpus_for_training \
                      --port $port_for_ddp_training
                      [--use_class] #add it into command for reaction class known setting
```

## Inference and evaluation

To inference the well-trained checkpoints, you can use the following command:

```shell
python inference.py --model_arch_path model_arch/uspto_50k.json \
                    --seed $random_seed \
                    --data_path $path_for_file_of_testset \
                    --device $device_id \
                    --checkpoint $path_of_checkpoint \
                    --token_ckpt $path_of_checkpoint_for_tokenizer \
                    --max_len $max_length_of_generated_smiles \
                    --beams $beam_size_for_beam_search \
                    --output_folder $the_folder_to_store_results \
                    --save_every $the_step_to_write_results_to_files \
                    --batch_size $batch_size_for_batched_inference \
                    --start $start_idx \
                    --len $num_of_samples_to_test \
                    --aug_time $num_of_smiles_augmentations \
                    [--disable_kv_cache] # add it to disable cached decoding \
                    [--use_class] # add it for reaction class known setting
```

The script writes results under the output folder using the `start-end.json` naming scheme. Use `--start 0 --len -1` to run the full file in one job, or split the file into multiple shards by changing `--start` and `--len`.

To evaluate a single output file and get top-k accuracy, use:

```shell
python evaluate_answer.py --beam $beam_size_for_beam_search --path $path_of_result --single_file
```

To evaluate a directory of sharded output files, use:

```shell
python evaluate_answer.py --beam $beam_size_for_beam_search --path $path_of_output_dir
```

We also provide the script for inferencing a single product. You can use the following command:

```shell
python inference_one.py --model_arch_path model_arch/uspto_50k.json \
                        --seed $random_seed \
                        --device $device_id \
                        --checkpoint $path_of_checkpoint \
                        --token_ckpt $path_of_checkpoint_for_tokenizer \
                        --max_len $max_length_of_generated_smiles \
                        --beams $beam_size_for_beam_search \
                        --product_smiles $the_SMILES_of_product \
                        --aug_time $num_of_smiles_augmentations \
                        --input_class $class_number_for_reaction \
                        [--disable_kv_cache] # add it to disable cached decoding \
                        [--use_class] # add it for reaction class known setting \
                        [--org_output] # add it to keep invalid smiles in outputs
```

If `--use_class` is added, `input_class` is required. Also make sure that the product SMILES contains a single molecule.

## Reproduction Launchers

We recently received several reproducibility issues. Because our servers were upgraded after the article was published, the launchers below do not exactly match the original hyperparameters used to train the released checkpoints. They provide practical parameter settings that can produce similar performance on the current server setup. These settings have not been exhaustively tuned and may still have room for optimization.

Hardware requirement:

- CUDA 11-compatible GPUs.
- USPTO-50K: GPUs with 48GB memory per card are required for the recommended launcher parameters.
- USPTO-MIT: GPUs with 48GB memory per card are required for the recommended launcher parameters.
- USPTO-FULL: GPUs with 80GB memory per card are required for the recommended launcher parameters.

**TODO before running: set `DATA_DIR` to your processed dataset folder.**

Replace `/path/to/USPTO-*` with the dataset folder itself, not its parent folder. The folder passed as `DATA_DIR` must directly contain:

```text
canonicalized_raw_train.csv
canonicalized_raw_val.csv
canonicalized_raw_test.csv
```

Usage:

```shell
# USPTO-50K
DATA_DIR=/path/to/USPTO-50K bash launchers/uspto-50k-train.sh all

# USPTO-MIT
DATA_DIR=/path/to/USPTO-MIT bash launchers/uspto-mit-train.sh all

# USPTO-FULL
DATA_DIR=/path/to/USPTO-FULL bash launchers/uspto-full-train.sh all
```

Available modes:

- USPTO-50K: `stage1`, `stage2_unknown`, `stage2_known`, `stage2`, `all`.
- USPTO-MIT: `stage1`, `stage2`, `all`.
- USPTO-FULL: `stage1`, `stage2`, `all`.

If no mode is provided, the launchers use `all` by default.

Optional flag: the current CLI name is `--scilence`. Add it only for non-interactive runs where progress bars should be disabled. We recommend not using it for normal training so that the progress bars remain visible.

The scripts do not embed any server-specific dataset path. If needed, override log folders, run names, checkpoints, token JSONs, or model architecture paths with the corresponding environment variables in the launcher.
