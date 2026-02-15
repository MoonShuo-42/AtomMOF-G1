#!/bin/bash

# =================================================================
# GLOBAL SETTINGS
# =================================================================
# Set your hardware resources here
GPUS=8
PROJECT_ROOT="/home/jovyan/AlphaMOF"
CHECKPOINT_ROOT="/home/jovyan/nayoung-ckpts"
PYTHON_SCRIPT="src/train.py"
BASE_DATA_CONF="data=bwdb"
LOGGER="logger=wandb"

# =================================================================
# MODEL CONFIGURATIONS
# =================================================================
# Medium (default)
TOKEN_S=512
TOKEN_Z=256
TOKEN_HEAD=8

# =================================================================
# STAGE CONFIGURATIONS
# =================================================================
BASE_NAME="vanilla_bwdb"

# --- Stage 1: Pretraining ---
STAGE_1_NAME="${BASE_NAME}"
STAGE_1_ATOMS=200
STAGE_1_SQUARE=1000000
STAGE_1_EPOCHS=500
STAGE_1_RESUME_CKPT="" # LEAVE EMPTY IF NOT RESUMING

# --- Stage 2: Fine-tuning 1 ---
STAGE_2_NAME="${BASE_NAME}_finetune-1"
STAGE_2_ATOMS=300
STAGE_2_SQUARE=1000000
STAGE_2_EPOCHS=100

# --- Stage 3: Fine-tuning 2 ---
STAGE_3_NAME="${BASE_NAME}_finetune-2"
STAGE_3_ATOMS="null"
STAGE_3_SQUARE=800000
STAGE_3_EPOCHS=100

# =================================================================
# HELPER FUNCTIONS
# =================================================================

# Function to find the latest checkpoint for a given task name
get_last_ckpt() {
    local task_name=$1
    local search_dir="${CHECKPOINT_ROOT}/${task_name}/runs/"

    # Sort reverse alphabetically to get the latest run
    local recent_run=$(ls -d "${search_dir}"* 2>/dev/null | sort -r | head -1)
    
    if [ -z "$recent_run" ]; then
        echo "Error: Could not find any runs in ${search_dir}"
        exit 1
    fi
    
    local ckpt_path="${recent_run}/checkpoints/last.ckpt"
    
    if [ ! -f "$ckpt_path" ]; then
        echo "Error: Checkpoint not found at ${ckpt_path}"
        exit 1
    fi
    
    echo "$ckpt_path"
}

run_stage() {
    local stage_num=$1
    local task_name=$2
    local atoms=$3
    local square=$4
    local epochs=$5
    local prev_task_name=$6 # Optional: for finding auto-ckpt from previous stage
    local manual_ckpt=$7    # [NEW] Optional: explicit path for resuming

    echo "----------------------------------------------------------------"
    echo "STARTING STAGE ${stage_num}: ${task_name}"
    echo "Config: Atoms=${atoms}, Square=${square}, Epochs=${epochs}, GPUs=${GPUS}"
    echo "----------------------------------------------------------------"

    # Base command construction
    local CMD="python ${PYTHON_SCRIPT} \
        ${BASE_DATA_CONF} \
        trainer.devices=${GPUS} \
        task_name=${task_name} \
        ${LOGGER} \
        +trainer.num_sanity_val_steps=0 \
        data.max_num_atoms=${atoms} \
        data.dynamic.max_num_atoms_square=${square} \
        trainer.max_epochs=${epochs} \
        model.training_args.lr_scheduler=af3 \
        model.token_s=${TOKEN_S} \
        model.token_z=${TOKEN_Z} \
        model.flow_model_args.token_transformer_heads=${TOKEN_HEAD}"

    # (Optional) Resume from ckpt for pretraining stage
    if [ ! -z "$manual_ckpt" ]; then
        echo "Resuming from manual checkpoint: ${manual_ckpt}"
        CMD="${CMD} resume_ckpt_path=${manual_ckpt}"
        
    # Detect checkpoint from previous stage
    elif [ ! -z "$prev_task_name" ]; then
        echo "Searching for checkpoint from previous stage: ${prev_task_name}..."
        local ckpt_path=$(get_last_ckpt "$prev_task_name")
        echo "Found checkpoint: ${ckpt_path}"
        CMD="${CMD} pretrained_ckpt_path=${ckpt_path}"
    fi

    # Execute the command
    echo "Executing: ${CMD}"
    eval $CMD
    
    # Check if the run was successful
    if [ $? -ne 0 ]; then
        echo "CRITICAL FAILURE: Stage ${stage_num} failed. Stopping pipeline."
        exit 1
    fi
    echo "Stage ${stage_num} completed successfully."
    echo ""
}

# =================================================================
# EXECUTION FLOW
# =================================================================

run_stage 1 "$STAGE_1_NAME" "$STAGE_1_ATOMS" "$STAGE_1_SQUARE" "$STAGE_1_EPOCHS" "" "$STAGE_1_RESUME_CKPT"
run_stage 2 "$STAGE_2_NAME" "$STAGE_2_ATOMS" "$STAGE_2_SQUARE" "$STAGE_2_EPOCHS" "$STAGE_1_NAME" ""
run_stage 3 "$STAGE_3_NAME" "$STAGE_3_ATOMS" "$STAGE_3_SQUARE" "$STAGE_3_EPOCHS" "$STAGE_2_NAME" ""

echo "----------------------------------------------------------------"
echo "FULL CURRICULUM PIPELINE COMPLETED SUCCESSFULLY"
echo "----------------------------------------------------------------"