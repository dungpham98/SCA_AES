#!/bin/bash

BATCH_SIZE=100
NUM_EPOCH=5
NAME_PREFIX="PhaseC_test"
MODEL_DIR="multi_attack_trained_models/PhaseB_test_phaseB"
EVAL_BASE="AES_PTv2_D"

for model_id in {10..29}; do
    MODEL_PATH="${MODEL_DIR}/model${model_id}.keras"
    
    for num_trace in 500 800; do
        for eval_id in {1..4}; do
            EVAL_PATH="${EVAL_BASE}${eval_id}.h5"
            RUN_NAME="${NAME_PREFIX}_model${model_id}_trace${num_trace}_D${eval_id}"
            
            echo "Running model${model_id}, trace=${num_trace}, eval=${EVAL_PATH}"
            python train_tuning.py \
                --batch_size $BATCH_SIZE \
                --num_epoch $NUM_EPOCH \
                --name $RUN_NAME \
                --train_model $MODEL_PATH \
                --num_sample 20 \
                --num_trace $num_trace \
                --eval_path $EVAL_PATH
        done
    done
done
