#!/bin/bash

# Function to limit the number of concurrent background jobs
wait_for_jobs() {
    local max_jobs=$1
    while true; do
        local current_jobs=$(jobs -p | wc -l)
        if [[ $current_jobs -lt $max_jobs ]]; then
            break
        fi
        sleep 1
    done
}

# Define intervals and scripts
intervals=("1w" "daily" "8h" "1h" )
# scripts=("main_nb.py")
scripts=("main.py")

# --------------- TRAINING PROCESSES ------------------

for seed in 3; do
    for interval in ${intervals[@]}; do
        for script in ${scripts[@]}; do
            # Configure directories based on the script being executed
            if [[ $script == "main.py" ]]; then
                data_dir="pth/crime_${interval}_origin_${seed}.pth"
                log_dir="log/crime_${interval}_origin_${seed}.txt"
            else
                data_dir="pth/crime_${interval}_nb_${seed}.pth"
                log_dir="log/crime_${interval}_nb_${seed}.txt"
            fi
            
            # Execute the script and launch the process in the background
            python $script --enable-cuda --using-data crime_$interval --data-dir $data_dir --log-dir $log_dir --seed $seed &
            
            # Wait until the number of concurrent jobs drops below the limit
            wait_for_jobs 2
        done
    done
done

# Wait for all background jobs to finish before proceeding to the testing processes
wait

# --------------- TESTING PROCESSES ------------------

# Update scripts for testing processes
# scripts=("test.py" "test_nb.py")
scripts=("test.py")

for seed in 3; do
    for interval in ${intervals[@]}; do
        for script in ${scripts[@]}; do
            # Configure directories based on the script being executed
            if [[ $script == "test.py" ]]; then
                data_dir="pth/crime_${interval}_origin_${seed}.pth"
                output_dir="output/crime_${interval}_origin_${seed}"
            else
                data_dir="pth/crime_${interval}_nb_${seed}.pth"
                output_dir="output/crime_${interval}_nb_${seed}"
            fi
            
            # Execute the script and launch the process in the background
            python $script --using-data crime_$interval --best-model $data_dir --output-dir $output_dir &
            
            # Wait until the number of concurrent jobs drops below the limit
            wait_for_jobs 2
        done
    done
done

# Wait for all background jobs to finish before ending the script
wait
