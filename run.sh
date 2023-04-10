#!/bin/bash
# Testing script; should get MRR@10 >= 42
set -exuo pipefail

MODEL='microsoft/MiniLM-L12-H384-uncased'

time python train_biencoder.py --model_name "${MODEL}" --max_queries 10000 --epochs 1 --train_batch_size 32

ls -t output | head -n1 | sed 's|^|output/|' | sed 's/$/ 10/' | xargs python eval_msmarco.py | tee -a output.txt