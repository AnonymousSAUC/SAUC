#!/bin/bash

declare -a arr=("demand_5min" "demand_15min" "demand_60min" "crime_1h" "crime_8h" "crime_daily" "crime_1w" "crash_1h" "crash_8h" "crash_daily" "crash_1w")

for i in "${arr[@]}"
do
   python generate_training_data.py --output_dir data/$i --traffic_df_filename data/$i.h5
done
