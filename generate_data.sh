
ROOT_DIR=/code/SpecForge
model_path=/code/predictions/qwen3_8B_baseline/qwen3_8B_full/model
echo ROOT_DIR
echo $ROOT_DIR
 
torchrun --nproc_per_node=1 \
    scripts/prepare_hidden_states.py \
    --model-path $model_path \
    --enable-aux-hidden-states \
    --data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --chat-template qwen \
    --max-length 2048 \
    --tp-size 1 \
    --batch-size 1 \
    --mem-frac=0.75 \
    --build-dataset-num-proc 1 \
    --num-samples 500
    # --num-samples 5 \