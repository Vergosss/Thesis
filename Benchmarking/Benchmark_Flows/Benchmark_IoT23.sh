#!/usr/bin/env bash
set -e 
#for i in {1..7}; do
#rm "Benchmark_${i}_GPUs.log"
#done

for i in {1..7}; do
CUDA_VISIBLE_DEVICES=$(seq -s "," 0 $((i-1))) python3 Benchmark_Inference_IoT23.py 2>&1 | tee "./../Results_IoT23/Benchmark_results_IoT23_${i}_GPUs.log"

sleep 3
done
