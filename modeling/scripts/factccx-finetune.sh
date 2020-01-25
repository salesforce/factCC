#! /bin/bash
# Fine-tuning BERT-base on CNN/DM data

# UPDATE PATHS BEFORE RUNNING SCRIPT
export CODE_PATH= # absolute path to modeling directory
export DATA_PATH= # absolute path to data directory
export OUTPUT_PATH= # absolute path to model checkpoint

export TASK_NAME=factcc_generated
export MODEL_NAME=bert-base-uncased

python3 $CODE_PATH/run.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_PATH \
  --model_type pbert \
  --model_name_or_path $MODEL_NAME \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 12 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --loss_lambda 0.1 \
  --evaluate_during_training \
  --eval_all_checkpoints \
  --overwrite_cache \
  --output_dir $OUTPUT_PATH/$MODEL_NAME-$TASK_NAME-finetune-$RANDOM/
