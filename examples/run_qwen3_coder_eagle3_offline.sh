SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

export CUDA_VISIBLE_DEVICES=0
# train eagle3 for qwen3-coder
NUM_GPUS=1
model_path=/code/predictions/qwen3_8B_baseline/qwen3_8B_full/model
    # --draft-model-config $ROOT_DIR/configs/qwen3-coder-480B-A35B-instruct-eagle3.json \
# dataset_path=/code/aijourney/eagle3/aij_judge_task_1_train_qwen3_8B_500.json
dataset_path=/code/aijourney/eagle3/aij_judge_task_1_train_qwen3_8B_10000.json
max_len=8196
output_path=qwen3-8b-Instruct_aij_10k
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_offline.py \
    --target-model-path $model_path \
    --train-data-path $dataset_path \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states \
    --output-dir $ROOT_DIR/outputs/$output_path \
    --num-epochs 10 \
    --draft-micro-batch-size 2 \
    --draft-global-batch-size 2 \
    --learning-rate 1e-4 \
    --max-length $max_len \
    --chat-template qwen \
    --draft-attention-backend sdpa
    # --resume


# qwen3-8b-Instruct_aij_10k
# Training Epoch 0: 100%|█████████████████████████████████████████████████████████| 5000/5000 [26:57<00:00,  3.09it/s, loss=0.13, acc=0.94]
# Training Epoch 1: 100%|█████████████████████████████████████████████████████████| 5000/5000 [26:57<00:00,  3.09it/s, loss=0.35, acc=0.73]
# Training Epoch 2: 100%|█████████████████████████████████████████████████████████| 5000/5000 [26:54<00:00,  3.10it/s, loss=0.49, acc=0.68]
# Training Epoch 3: 100%|█████████████████████████████████████████████████████████| 5000/5000 [26:42<00:00,  3.12it/s, loss=0.49, acc=0.69]
# Training Epoch 4: 100%|█████████████████████████████████████████████████████████| 5000/5000 [26:41<00:00,  3.12it/s, loss=0.22, acc=0.85]
# Training Epoch 5: 100%|█████████████████████████████████████████████████████████| 5000/5000 [26:53<00:00,  3.10it/s, loss=0.21, acc=0.89]
# Training Epoch 6: 100%|█████████████████████████████████████████████████████████| 5000/5000 [26:48<00:00,  3.11it/s, loss=0.22, acc=0.80]
# Training Epoch 7: 100%|█████████████████████████████████████████████████████████| 5000/5000 [26:42<00:00,  3.12it/s, loss=0.32, acc=0.78]
# Training Epoch 8: 100%|█████████████████████████████████████████████████████████| 5000/5000 [26:37<00:00,  3.13it/s, loss=0.24, acc=0.87]
# Training Epoch 9: 100%|█████████████████████████████████████████████████████████| 5000/5000 [26:42<00:00,  3.12it/s, loss=0.18, acc=0.90]
