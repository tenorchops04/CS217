#!/bin/bash

for i in {1..5}
do
    echo "Running iteration $i:" >> results.txt
    python3 ./image_classifiers.py --sg >> results.txt
    rm -rf dataset/neurips17_dataset/batch_result_0.05_pm_2/
    python3 ./image_classifiers.py --mg >> results.txt
    rm -rf dataset/neurips17_dataset/batch_result_0.05_pm_2/
done