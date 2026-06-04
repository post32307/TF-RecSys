#!/usr/bin/env bash

# Directory containing datasets (assumes this script is in the code directory)
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../data" && pwd)"

# Seeds to sweep
SEEDS=(2020)

# Create an array of GPU indices
GPUS=(0 2 3 4)
# Counter for assigning GPUs
COUNTER=0

# -------------------------------------------------
# Pre-calculate total experiments to show percentage
# -------------------------------------------------
NUM_DATASETS=0
for ds in "$DATA_DIR"/*; do
  if [[ -d "$ds" ]]; then
    ((NUM_DATASETS++))
  fi
done

TOTAL_EXPERIMENTS=$(( NUM_DATASETS * ${#SEEDS[@]} ))

if (( TOTAL_EXPERIMENTS == 0 )); then
  echo "No valid datasets found or seeds array is empty. Exiting."
  exit 0
fi

# -------------------------------------------------
# Signal handling – stop all background jobs on Ctrl+C
cleanup() {
  echo "Caught SIGINT/SIGTERM – terminating all background processes..."
  
  # 'jobs -p' outputs the PIDs of all background jobs managed by this shell,
  # catching the python process rather than just the tail end of the pipe.
  local job_pids=$(jobs -p)
  
  if [[ -n "$job_pids" ]]; then
    # Kill all of them, suppressing errors if they've already exited cleanly
    kill -SIGTERM $job_pids 2>/dev/null || true
  fi
  
  wait
  exit 1
}
trap cleanup SIGINT SIGTERM
# -------------------------------------------------

ENV_NAME="lgn"
MAX_JOBS=8 #8  # Maximum concurrent jobs

# Initialize conda properly for bash scripts, then activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Create logs directory if it doesn't exist
mkdir -p logs

# Loop over each dataset directory
for ds in "$DATA_DIR"/*; do
  if [[ -d "$ds" ]]; then
    dataset=$(basename "$ds")
    for seed in "${SEEDS[@]}"; do
        
        # Determine GPU for this job (round-robin)
        GPU_IDX=${GPUS[$((COUNTER % ${#GPUS[@]}))]}
        COUNTER=$((COUNTER + 1))
        
        # Calculate and display the percentage
        PERCENT=$(( COUNTER * 100 / TOTAL_EXPERIMENTS ))
        echo "[${PERCENT}%] (${COUNTER}/${TOTAL_EXPERIMENTS}) Running dataset=$dataset seed=$seed on GPU $GPU_IDX"
        
        # Run the job in the background, redirecting ALL output silently to the log file
        python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/main.py" \
            --dataset "$dataset" \
            --gpu_ID "$GPU_IDX" \
            --seed "$seed" >> logs/${dataset}_${seed}.txt 2>&1 &
        
        # Control number of background jobs
        while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
          sleep 1
        done
        
    done
  fi
done

# Wait for all background jobs to finish
wait

echo "All ${TOTAL_EXPERIMENTS} sweeps completed."