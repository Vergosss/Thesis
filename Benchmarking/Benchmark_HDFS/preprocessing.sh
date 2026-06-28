#!/usr/bin/env bash
set -e
for i in {1..10}; do
python3 preprocessing.py $i 2>&1 | tee "./Preprocessing_logs/preprocessing_HDFS_${i}_workers.log"
sleep 3
done
