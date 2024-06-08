# UAlign

Official implementation for paper:

[UAlign: Pushing the Limit of Template-free Retrosynthesis Prediction with Unsupervised SMILES Alignment](https://arxiv.org/abs/2404.00044)

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
    - Every checkpoint needs to be used together with its corresponding tokenizer. The tokenizers are stored as $\texttt{pkl}$ files, while the trained model weights are stored in $\texttt{pth}$​ files. The matching model weights and tokenizer have the same name and are placed in the same folder.
    
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
      
    
- predicted_results
    - In the $\texttt{USPTO-FULL}$ and $\texttt{USPTO-MIT}$ folders, there is only one set of experimental results in each. They are divided into different files based on the index of the data. 
    - In USPTO-50K, there are two sets of experimental results. The file $\texttt{answer-1711345166.9484136.json}$ corresponds to the setting of reaction class unknown, while $\texttt{answer-1711345359.2533984.json}$ corresponds to the setting of reaction class known. 
    - Each $\texttt{json}$ file contains raw data for testing, the model's prediction results, corresponding logits, and also includes the checkpoints information used to generate this $\texttt{json}$ file.



## Data Preprocess

We provide the data preprocess scripts in folder $\texttt{data\_proprocess}$​. Each dataset is processed through a separate processing script.  The atom-mapping numbers of each reaction are reassigned according to the canonical ranks of atoms of the product to avoid information leakage. The script for USPTO-50K and USPTO-FULL is used to process a single file. The scripts can be used as follows and the output file will be stored in the same folder as the input file.

```shell
python data_proprocess/canonicalize_data_50k.py --filename $dir_of_raw_file
python data_proprocess/canonicalize_data_full.py --filename $dir_of_raw_file
```

The script for USPTO-MIT processes all the files together, which can be used by

```shell
python data_proprocess/canonicalize_data_full.py --dir $folder_of_raw_data --output_dir $output_dir
```

The `$folder_of_raw_data` should contain the following files:  $\texttt{train.txt}$, $\texttt{valid.txt}$ and $\texttt{test.txt}$​. 

**For the detail about data preprocess, please refer to the article.**

## Generating Tokens

To build the tokenizer, we need a list of of all the shown tokens. You can use the follow command to generate the token list and store it in files.

```
python generate_tokens $file_1 $file_2 ... $file_n $token_list.json
```

The script can accept multiple files as input and the last position should be the path of file to store the token list. The files should have the same format as the processed dataset.

## Stage I training

Use the following command for training the first stage:

```shell
python pretrain.py --dim $dim \
				   --n_layer $n_layer \
                   --data_path $folder_of_dataset \
                   --seed $random_seed \
                   --bs $batch_size \
                   --epoch $epoch_for_training \
                   --early_stop $epoch_num_for_checking_early_stop \
                   --device $device_id \
                   --lr $learning_rate \
                   --dropout $dropout \
                   --base_log $folder_for_logging \
                   --heads $num_heads_for_attention \
                   --negative_slope $negative_slope_for_leaky_relu \
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
python ddp_pretrain.py --dim $dim \
				   	   --n_layer $n_layer \
                       --data_path $folder_of_dataset \
                       --seed $random_seed \
                       --bs $batch_size \
                       --epoch $epoch_for_training \
                       --early_stop $epoch_num_for_checking_early_stop \
                       --lr $learning_rate \
                       --dropout $dropout \
                       --base_log $folder_for_logging \
                       --heads $num_heads_for_attention \
                       --negative_slope $negative_slope_for_leaky_relu \
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
python train_trans.py --dim $dim \
                          --n_layer $n_layer \
                          --aug_prob $probability_for_data_augumentation \
                          --data_path $folder_of_dataset \
                          --seed $random_seed \
                          --bs $batch_size \
                          --epoch $epoch_for_training \
                          --early_stop $epoch_num_for_checking_early_stop \
                          --lr $learning_rate \
                          --dropout $dropout \
                          --base_log $folder_for_logging \
                          --heads $num_heads_for_attention \
                          --negative_slope $negative_slope_for_leaky_relu \
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
python ddp_train_trans.py --dim $dim \
				      --n_layer $n_layer \
				      --aug_prob $probability_for_data_augumentation \
				      --data_path $folder_of_dataset \
                      --seed $random_seed \
                      --bs $batch_size \
                      --epoch $epoch_for_training \
                      --early_stop $epoch_num_for_checking_early_stop \
                      --device $device_id \
                      --lr $learning_rate \
                      --dropout $dropout \
                      --base_log $folder_for_logging \
                      --heads $num_heads_for_attention \
                      --negative_slope $negative_slope_for_leaky_relu \
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

To inference the well-trained checkpoints, you can use the following commands:

```shell
python inference.py --dim $dim \
                    --n_layer $n_layer \
                    --heads $num_heads_for_attention \
                    --seed $random_seed \
                    --data_path $path_for_file_of_testset \
                    --device $device_id \
                    --checkpoint $path_of_checkpoint \
                    --token_ckpt $path_of_checkpoint_for_tokenizer \
                    --negative_slope $negative_slope_for_leaky_relu \
                    --max_len $max_length_of_generated_smiles \
                    --beams $beam_size_for_beam_search \
                    --output_folder $the_folder_to_store_results \
                    --save_every $the_step_to_write_results_to_files \
                    [--use_class] #add it into command for reaction class known setting
```

The script will summary all the results into a $\texttt{json}$ file under output folder, named by the timestamp. And to evaluate the result to get top-$k$ accuracy, use   the following command:

```shell
python evaluate_answer.py --beams $beam_size_for_beam_search --path $path_of_result
```

To fasten the inference, you can the following command to inference only a part of test set so that the inference part can be done parallelly: 

```shell
python inference_part.py --dim $dim \
                    	 --n_layer $n_layer \
                         --heads $num_heads_for_attention \
                         --seed $random_seed \
                         --data_path $path_for_file_of_testset \
                         --device $device_id \
                         --checkpoint $path_of_checkpoint \
                         --token_ckpt $path_of_checkpoint_for_tokenizer \
                         --negative_slope $negative_slope_for_leaky_relu \
                         --max_len $max_length_of_generated_smiles \
                         --beams $beam_size_for_beam_search \
                         --output_folder $the_folder_to_store_results \
                         --save_every $the_step_to_write_results_to_files \
                         --start $start_idx \
                         --len $num_of_samples_to_test \
                         [--use_class] #add it into command for reaction class known setting
```

The script will summary all the results into a $\texttt{json}$ file under output folder, named by the start and end index of data. And to evaluate the result to get top-$k$​ accuracy, use the following command:

```shell
python evaluate_dir.py --beams $beam_size_for_beam_search --path $path_of_output_dir
```

We also provide the script for inferencing a single product. You can used the following command for inference:

```shell
python inference_one.py --dim $dim \
                        --n_layer $n_layer \
                    	--heads $num_heads_for_attention \
                    	--seed $random_seed \
                    	--device $device_id \
                    	--checkpoint $path_of_checkpoint \
                    	--token_ckpt $path_of_checkpoint_for_tokenizer \
                    	--negative_slope $negative_slope_for_leaky_relu \
                    	--max_len $max_length_of_generated_smiles \
                    	--beams $beam_size_for_beam_search \
                    	--product_smiles $the_SMILES_of_product \
                    	--input_class $class_number_for_reaction \
                        [--use_class] #add it into command for reaction class known setting
                        [--org_output] # add it and the invalid smiles will not be removed from outputs
```

If `--use_class` is added, the `input_class` is required. Also you have make sure that the product SMILES contains a single molecule. 

