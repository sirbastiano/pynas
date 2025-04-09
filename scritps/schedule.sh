#!/bin/bash

LOG_FILE="scheduler.log"
COMMANDS=(
    "conda activate pynas && python3 /Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/nas_seg_burned.py --mutation_probability 0.1"
    "conda activate pynas && python3 /Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/nas_seg_burned.py --mutation_probability 0.2"
    "conda activate pynas && python3 /Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/nas_seg_burned.py --mutation_probability 0.3"
)

MAX_RETRIES=5
PARALLEL_JOBS=2  # Adjust this based on your GPU capacity

# Function to run a command and retry if it fails
run_command() {
    local cmd="$1"
    local retries=0

    while [ $retries -lt $MAX_RETRIES ]; do
        echo "Running: $cmd" | tee -a "$LOG_FILE"
        eval "$cmd" >>"$LOG_FILE" 2>&1
        if [ $? -eq 0 ]; then
            echo "Success: $cmd" | tee -a "$LOG_FILE"
            return 0
        else
            echo "Failed: $cmd (Retry $((retries + 1))/$MAX_RETRIES)" | tee -a "$LOG_FILE"
            retries=$((retries + 1))
        fi
    done

    echo "Giving up: $cmd after $MAX_RETRIES retries" | tee -a "$LOG_FILE"
    return 1
}

# Export the function so it can be used by parallel
export -f run_command
export LOG_FILE
export MAX_RETRIES

# Run commands in parallel
echo "Starting parallel execution..." | tee -a "$LOG_FILE"
printf "%s\n" "${COMMANDS[@]}" | xargs -n 1 -P $PARALLEL_JOBS -I {} bash -c 'run_command "{}"'
echo "All tasks completed." | tee -a "$LOG_FILE"