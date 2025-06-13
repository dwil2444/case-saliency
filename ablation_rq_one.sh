#!/bin/bash
set -e


MODEL_NAME=${1:-resnet}
THRESHOLD=${2:-70}
K=${3:-5}

NUM_RUNS=5
DATA_ROOT="./data"
OUTPUT_DIR="./.ablation/sensitivity_study"
CLASS_FILE="/home/dw3zn/Desktop/Repos/CASE/load_datasets/cifar100_labels.txt"
VENV_PATH="venv"
CUDA_DEVICE=0


# Activate virtual environment if it exists
if [ -d "$VENV_PATH" ]; then
  echo "[INFO] Activating virtual environment: $VENV_PATH"
  source "$VENV_PATH/bin/activate"
else
  echo "[WARN] No virtual environment found at $VENV_PATH. Continuing without activation."
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
echo "[INFO] Using CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"


mkdir -p "$OUTPUT_DIR"

# 1. Loop over N runs
for SEED in $(seq 0 $((NUM_RUNS - 1))); do
    RUN_DIR="${OUTPUT_DIR}/run_${SEED}"
    MODEL_PATH="${RUN_DIR}/${MODEL_NAME}_best.pth"
    AGREEMENT_CSV="${RUN_DIR}/agreement.csv"
    STDOUT_LOG="${RUN_DIR}/run.log"
    STDERR_LOG="${RUN_DIR}/run.err"
    mkdir -p "$RUN_DIR"

    echo "[Run $SEED] Training model..."
    python -m ablation.train_model \
        --model_name=$MODEL_NAME \
        --data_root=$DATA_ROOT \
        --output_path=$MODEL_PATH \
        --batch_size 32 \
        --epochs 20 \
        --seed=$SEED > "$STDOUT_LOG" 2> "$STDERR_LOG"

    echo "[Run $SEED] Computing Agreement Metrics"
    python -m ablation.agreement_finetuned \
    --model_path=$MODEL_PATH \
    --model_name=$MODEL_NAME \
    --k=$K \
    --data_root=$DATA_ROOT \
    --class_file=$CLASS_FILE \
    --output_csv=$AGREEMENT_CSV \
    --num_samples=1000 >> "$STDOUT_LOG" 2>> "$STDERR_LOG"

    echo "[Run $SEED] Running Statistical Analysis"
    python -m plots.rq_one_analysis \
    --csv_path=$AGREEMENT_CSV \
    --output_dir=$RUN_DIR \
    --threshold=$THRESHOLD >> "$STDOUT_LOG" 2>> "$STDERR_LOG"


done


# Next - add evaluation of each trained model
# i.e. get explanations for the model on each XAI method
# then do the sensitivity analysis on the csv for each model instance

echo "✅ All done. Results in ${OUTPUT_DIR}/analysis"
