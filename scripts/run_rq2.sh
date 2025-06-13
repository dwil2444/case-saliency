#!/bin/bash
set -e

# -----------------------------
# Configurable inputs
MODEL=${1:-resnet}           # default: 'resnet'
K=${2:-50}                   # default: 50
NUM_SAMPLES=${3:-500}        # default: 500
CONTRAST_K=${4:-1}

# Resolve base directory relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths and fixed configuration
DATA_ROOT="${WORKING_DIR}/data/imagenet"
CLASS_FILE="${WORKING_DIR}/load_datasets/imagenet_classes.txt"
OUTPUT_DIR="${WORKING_DIR}/.rqtwo"
OUTPUT_CSV="${OUTPUT_DIR}/saliency_eval.csv"
STDOUT_LOG="${OUTPUT_DIR}/run.log"
STDERR_LOG="${OUTPUT_DIR}/run.err"
VENV_PATH="${WORKING_DIR}/venv"
CUDA_DEVICE=0
# -----------------------------

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

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Print run info
echo "[INFO] Running evaluation"
echo "MODEL       = $MODEL"
echo "K           = $K"
echo "NUM_SAMPLES = $NUM_SAMPLES"
echo "OUTPUT_DIR  = $OUTPUT_DIR"
echo "CONTRAST_K  = $CONTRAST_K"

# Step 1: Run evaluation
python -m CAM_eval.comprehensive_eval \
    --model_name=$MODEL \
    --k=$K \
    --data_root=$DATA_ROOT \
    --class_file=$CLASS_FILE \
    --output_csv=$OUTPUT_CSV \
    --num_samples=$NUM_SAMPLES \
    --contrast_k=$CONTRAST_K \
    > "$STDOUT_LOG" 2> "$STDERR_LOG"

# Step 2: Run post-analysis
python -m plots.analysis \
    --csv_path=$OUTPUT_CSV \
    --output_dir=$OUTPUT_DIR \
    --reference_method="case" \
    --save_plots=True \
    >> "$STDOUT_LOG" 2>> "$STDERR_LOG"
