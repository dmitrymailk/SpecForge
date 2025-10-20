
ROOT_DIR=/code/SpecForge
# model_path=/code/predictions/qwen3_8B_baseline/qwen3_8B_full/model
model_path=/code/model_checkpoints/quantization/Qwen3-8B-AWQ
echo ROOT_DIR
echo $ROOT_DIR
max_len=16384
# dataset_path=/code/aijourney/eagle3/aij_judge_task_1_train_qwen3_8B_500.json
dataset_path=/code/aijourney/eagle3/aij_judge_task_1_train_qwen3_8B_10000.json
torchrun --nproc_per_node=1 \
    scripts/prepare_hidden_states.py \
    --model-path $model_path \
    --enable-aux-hidden-states \
    --data-path $dataset_path \
    --chat-template qwen \
    --max-length $max_len \
    --tp-size 1 \
    --batch-size 1 \
    --mem-frac=0.75 \
    --build-dataset-num-proc 1 \
    --num-samples 10000
    # --num-samples 500
    # --num-samples 5 \