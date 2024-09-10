#!/bin/bash

# GPU config
USE_TPU=False

# Experiment (data/checkpoint/directory) config
DATA_PATH=#Path to the .h5 file containing the dataset
DATASET=ASCAD
CKP_DIR=./
CKP_IDX=0
WARM_START=False
RESULT_PATH=results

# Optimization config
LEARNING_RATE=2.5e-4
CLIP=0.25
MIN_LR_RATIO=0.004
INPUT_LENGTH=98000 # Trace length or size of the attack window
PROF_DESYNC=600 # Profiling Desync; it will introduce random desync 
                # in [0, $PROF_DESYNC) in the profiling traces at the begining
ATTACK_DESYNC=600 # Attack Desync; it will introduce random desync 
                # in [0, $ATTACK_DESYNC) in the attack traces at the begining
DATA_DESYNC=400 # Max desync for data augmentation; it will introduce an extra 
		# random desync in [0, $DATA_DESYNC) in the profiling traces 
		# at the begin of each epoch during training
START_IDX=0 # Starting index of the attack window; should be zero for
            # attacking the full-length traces

# Training config
TRAIN_BSZ=16
EVAL_BSZ=16
TRAIN_STEPS=4000000
WARMUP_STEPS=1000000
ITERATIONS=20000
SAVE_STEPS=40000

# Model config
N_LAYER=2
D_MODEL=128
D_HEAD=32
N_HEAD=8
D_INNER=256
N_HEAD_SM=2
D_HEAD_SM=64
DROPOUT=0.05
CONV_KERNEL_SIZE=7
N_CONV_LAYER=2
POOL_SIZE=6
D_KERNEL_MAP=512
BETA_HAT_2=150
MODEL_NORM='preLC'
HEAD_INIT='forward'
SEG_LEN=5000 # Segment length
SEG_STRIDE=3000 # Stride length for creating new segment; overlap
                # between two consequetive segments is 
		# $SEG_LEN-$SEG_STRIDE = 2000
SM_ATTN=True

# Evaluation config
MAX_EVAL_BATCH=1
OUTPUT_ATTN=False


if [[ $1 == 'train' ]]; then
    python train_trans.py \
        --use_tpu=${USE_TPU} \
        --data_path=${DATA_PATH} \
	--dataset=${DATASET} \
        --checkpoint_dir=${CKP_DIR} \
        --warm_start=${WARM_START} \
        --result_path=${RESULT_PATH} \
        --learning_rate=${LEARNING_RATE} \
        --clip=${CLIP} \
        --min_lr_ratio=${MIN_LR_RATIO} \
        --warmup_steps=${WARMUP_STEPS} \
	--input_length=${INPUT_LENGTH} \
	--prof_desync=${PROF_DESYNC} \
	--attack_desync=${ATTACK_DESYNC} \
	--data_desync=${DATA_DESYNC} \
	--start_idx=${START_IDX} \
        --train_batch_size=${TRAIN_BSZ} \
        --eval_batch_size=${EVAL_BSZ} \
        --train_steps=${TRAIN_STEPS} \
        --iterations=${ITERATIONS} \
        --save_steps=${SAVE_STEPS} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
	--d_head=${D_HEAD} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --n_head_softmax=${N_HEAD_SM} \
        --d_head_softmax=${D_HEAD_SM} \
        --dropout=${DROPOUT} \
        --dropatt=${DROPATT} \
        --conv_kernel_size=${CONV_KERNEL_SIZE} \
	--n_conv_layer=${N_CONV_LAYER} \
        --pool_size=${POOL_SIZE} \
	--d_kernel_map=${D_KERNEL_MAP} \
	--beta_hat_2=${BETA_HAT_2} \
	--model_normalization=${MODEL_NORM} \
	--head_initialization=${HEAD_INIT} \
	--seg_len=${SEG_LEN} \
	--seg_stride=${SEG_STRIDE} \
	--softmax_attn=${SM_ATTN} \
        --max_eval_batch=${MAX_EVAL_BATCH} \
	--do_train=True
elif [[ $1 == 'test' ]]; then
    python train_trans.py \
        --use_tpu=${USE_TPU} \
        --data_path=${DATA_PATH} \
	--dataset=${DATASET} \
        --checkpoint_dir=${CKP_DIR} \
	--checkpoint_idx=${CKP_IDX} \
        --warm_start=${WARM_START} \
        --result_path=${RESULT_PATH} \
        --learning_rate=${LEARNING_RATE} \
        --clip=${CLIP} \
        --min_lr_ratio=${MIN_LR_RATIO} \
        --warmup_steps=${WARMUP_STEPS} \
	--input_length=${INPUT_LENGTH} \
	--prof_desync=${PROF_DESYNC} \
	--attack_desync=${ATTACK_DESYNC} \
	--start_idx=${START_IDX} \
        --train_batch_size=${TRAIN_BSZ} \
        --eval_batch_size=${EVAL_BSZ} \
        --train_steps=${TRAIN_STEPS} \
        --iterations=${ITERATIONS} \
        --save_steps=${SAVE_STEPS} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
	--d_head=${D_HEAD} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --n_head_softmax=${N_HEAD_SM} \
        --d_head_softmax=${D_HEAD_SM} \
        --dropout=${DROPOUT} \
        --dropatt=${DROPATT} \
        --conv_kernel_size=${CONV_KERNEL_SIZE} \
	--n_conv_layer=${N_CONV_LAYER} \
        --pool_size=${POOL_SIZE} \
	--d_kernel_map=${D_KERNEL_MAP} \
	--beta_hat_2=${BETA_HAT_2} \
	--model_normalization=${MODEL_NORM} \
	--head_initialization=${HEAD_INIT} \
	--seg_len=${SEG_LEN} \
	--seg_stride=${SEG_STRIDE} \
	--softmax_attn=${SM_ATTN} \
        --max_eval_batch=${MAX_EVAL_BATCH} \
	--output_attn=${OUTPUT_ATTN} \
	--do_train=False
else
    echo "unknown argument 1"
fi
