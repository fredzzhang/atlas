cd ..

TASK=$1
CHECKPOINT=$2
IDENTIFIER=$3
GPU=$4
SHOTS=1

IDENTIFIER=${TASK}_${IDENTIFIER}
echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=$GPU \
python tune_hps_singletask_taskvectors.py \
--task_dir data_more_shots/${SHOTS}_shot/${TASK}/ \
--checkpoints models/commonsense_qa-lora/1_shot/singletask-commonsense_qa/commonsense_qa_1_13_0.002_8_best-model.pt models/quail-lora/1_shot/singletask-quail/quail_1_13_0.002_4_best-model.pt models/sciq-lora/1_shot/singletask-sciq/sciq_1_13_0.002_4_best-model.pt \
--do_predict \
--learning_rate_list 2e-3 \
--bsz_list 2 4 8 \
--eval_period 20 \
--warmup_steps 50 \
--max_grad_norm 0.1 \
--weight_decay 0.01 \
--model google/flan-t5-large \
--output_dir models/${IDENTIFIER}/${SHOTS}_shot/singletask-${TASK} \
--gradient_accumulation_steps 1 \
--predict_batch_size 8 \
--num_train_epochs 300 \
--tasks sciq quail commonsense_qa;