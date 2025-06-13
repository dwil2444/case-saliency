#!/bin/bash
set -e

# -----------------------------
# Configurable inputs
MODEL=${1:-resnet}           # default: 'resnet'
K=${2:-5}                    # default: 5 (%)
NUM_SAMPLES=${3:-500}       # default: 500
THRESHOLD=${4:-70}          # default: 70 (% agreement for H0)

# Resolve base directory relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths and fixed configuration
DATA_ROOT="${WORKING_DIR}/data/imagenet"
CLASS_FILE="${WORKING_DIR}/load_datasets/imagenet_classes.txt"
OUTPUT_DIR="${WORKING_DIR}/.rqone/rq1_${MODEL}_k${K}"
OUTPUT_CSV="${OUTPUT_DIR}/agreement_eval.csv"
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
echo "[INFO] Running class-sensitivity evaluation for RQ1"
echo "MODEL       = $MODEL"
echo "K (%)       = $K"
echo "NUM_SAMPLES = $NUM_SAMPLES"
echo "THRESHOLD   = $THRESHOLD"
echo "OUTPUT_DIR  = $OUTPUT_DIR"

# Step 1: Run agreement extraction
python -m CAM_eval.class_sensitive \
    --model_name=$MODEL \
    --k=$K \
    --data_root=$DATA_ROOT \
    --class_file=$CLASS_FILE \
    --output_csv=$OUTPUT_CSV \
    --num_samples=$NUM_SAMPLES \
    > "$STDOUT_LOG" 2> "$STDERR_LOG"

# Step 2: Run statistical analysis
python -m plots.rq_one_analysis \
    --csv_path=$OUTPUT_CSV \
    --output_dir=$OUTPUT_DIR \
    --threshold=$THRESHOLD \
    >> "$STDOUT_LOG" 2>> "$STDERR_LOG"
