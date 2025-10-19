SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

export CUDA_VISIBLE_DEVICES=0
# train eagle3 for qwen3-coder
NUM_GPUS=1
model_path=/code/predictions/qwen3_8B_baseline/qwen3_8B_full/model
    # --draft-model-config $ROOT_DIR/configs/qwen3-coder-480B-A35B-instruct-eagle3.json \
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_offline.py \
    --target-model-path $model_path \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-Instruct \
    --num-epochs 10 \
    --draft-micro-batch-size 2 \
    --draft-global-batch-size 2 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    # --resume
