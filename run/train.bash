name=large_leaderboard_bs4_grad_accu_4
code_dir=src
flag="--do_train 
      --do_eval
      --evaluate_during_training
      --overwrite_output_dir
      --gradient_accumulation_steps 4
      --num_train_epochs 8
      --warmup_steps 150
      --logging_steps 500
      --save_steps 5000
      --max_seq_length 150
      --per_gpu_train_batch_size 4
      --learning_rate 1e-5
      --data_dir ./data
      --model_type roberta
      --model_name_or_path roberta-large
      --task_name piqa"


mkdir -p snap/$name
# cp $0 snap/$name/.bash
cp $0 snap/$name/train.bash
# clean previous code if any
rm -rf snap/$name/code
cp -r $code_dir snap/$name/code


# CUDA_VISIBLE_DEVICES=$1 python src/train.py $flag --output_dir ./snap/$name
CUDA_VISIBLE_DEVICES=$1 python -u src/train.py $flag --output_dir ./snap/$name | tee snap/$name/log

