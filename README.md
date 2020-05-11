# Branch for mctaco


## train

### bert baseline: 

`cd src`

`CUDA_VISIBLE_DEVICES=1 python train.py --task_name mctaco --model_type mctaco-bert --train_file dev_3783.tsv --val_file test_9442.tsv  --data_dir ../<your-data-folder> --model_name_or_path bert-base-uncased --do_lower_case --evaluate_during_training --do_train --do_eval --max_seq_length 128 --per_gpu_train_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 2e-5 --num_train_epochs 3 --warmup_steps 150 --output_dir ../snap/mctaco-bert-baseline-debug0`




### bert BCE:

`cd src`

`CUDA_VISIBLE_DEVICES=1 python train.py --task_name mctaco-bce --model_type mctaco-bert --model_name_or_path bert-base-uncased --do_lower_case --evaluate_during_training --train_file dev_3783.tsv --val_file test_9442.tsv  --data_dir ../mctaco-data --do_train --do_eval --max_seq_length 128 --per_gpu_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 2e-5 --num_train_epochs 3 --warmup_steps 150 --output_dir ../snap/mctaco-bert-bce-debug0`

(change to `--model_type mctaco-roberta` for roberta baseline and bce)

## evaluate

`python evaluator/evaluator.py eval --test_file <your-data-folder>/test_9442.tsv --prediction_file snap/mctaco-roberta-bce-ep3-2/eval_outputs.txt`