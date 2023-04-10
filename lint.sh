#!/bin/bash
set -exuo pipefail

isort *.py

pyflakes *.py
mypy train_biencoder.py