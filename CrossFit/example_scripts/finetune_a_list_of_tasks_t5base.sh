cd ..

TASKS=$1
CHECKPOINT=$2
IDENTIFIER=$3
GPU=$4

for SHOTS in 1 2 3 4
do
    for TASK in commonsense_qa quail sciq
    do
    IDENTIFIER=${TASK}-lora
    echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"

    CUDA_VISIBLE_DEVICES=$GPU \
    python tune_hps_singletask.py \
    --task_dir data_more_shots/${SHOTS}_shot/${TASK}/ \
    --checkpoint $CHECKPOINT \
    --do_train \
    --do_predict \
    --learning_rate_list 2e-3\
    --bsz_list 2 4 8 \
    --eval_period 20 \
    --warmup_steps 50 \
    --max_grad_norm 0.1 \
    --weight_decay 0.01 \
    --model google/flan-t5-large \
    --output_dir models/${IDENTIFIER}/${SHOTS}_shot/singletask-${TASK} \
    --gradient_accumulation_steps 1 \
    --predict_batch_size 8 \
    --num_train_epochs 300;

    done
done