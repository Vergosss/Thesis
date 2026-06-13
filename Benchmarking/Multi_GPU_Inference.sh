#!usr/bin/env bash

#Run The inference experiment for all combinations of availiable GPUs with accelerate. Log results into a file
for i in {1..7}; do

accelerate launch --config_file "Benchmark_config.yaml" "Benchmark_Inference.py" --num_processes="${i}" 2>&1 | tee "Benchmark_${i}_GPUs.log"

done

#Run these inside a tmux session to monitor the experiments' progress. Each experiment runs isolated and all experiments
#Run sequentially to prevent experiments affecting each other's results.