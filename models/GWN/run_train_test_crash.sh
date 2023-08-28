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
scripts=("train.py" "train_nb.py")

# --------------- TRAINING PROCESSES ------------------

for seed in 1; do
    for interval in ${intervals[@]}; do
        for script in ${scripts[@]}; do
            # Configure directories based on the script being executed
            if [[ $script == "train.py" ]]; then
                data_dir="pth/crash_${interval}_origin"
                log_dir="log/crash_${interval}_origin_${seed}.txt"
            else
                data_dir="pth/crash_${interval}_nb"
                log_dir="log/crash_${interval}_nb_${seed}.txt"
            fi
            
            # Execute the script and launch the process in the background
            python $script --data data/crash_$interval --num_nodes 277 --adjdata data/sensor_graph/crash_adj_mx_unix.pkl --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --batch_size 24 --seed 1 --save ./garage/crash_${interval}_1 --expid 1 --data-dir $data_dir --log-dir $log_dir --seed $seed &
            # Wait until the number of concurrent jobs drops below the limit
            wait_for_jobs 2
        done
    done
done

# Wait for all background jobs to finish before proceeding to the testing processes
wait

# --------------- TESTING PROCESSES ------------------

# Update scripts for testing processes
scripts=("test.py" "test_nb.py")

for seed in 1; do
    for interval in ${intervals[@]}; do
        for script in ${scripts[@]}; do
            # Configure directories based on the script being executed
            if [[ $script == "test.py" ]]; then
                ckpt_dir="pth/crash_${interval}_origin_exp1_best.pth"
                output_dir="output/crash_${interval}_origin_${seed}.npz"
            else
                ckpt_dir="pth/crash_${interval}_nb_exp1_best.pth"
                output_dir="output/crash_${interval}_nb_${seed}.npz"
            fi
            
            # Execute the script and launch the process in the background
            python $script --data data/crash_$interval --num_nodes 277 --adjdata data/sensor_graph/crash_adj_mx_unix.pkl --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --batch_size 24 --save $output_dir --checkpoint $ckpt_dir
            # Wait until the number of concurrent jobs drops below the limit
            wait_for_jobs 2
        done
    done
done

# Wait for all background jobs to finish before ending the script
wait
