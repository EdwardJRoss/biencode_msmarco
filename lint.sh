#!/bin/bash
set -exuo pipefail

pyflakes *.py
mypy train_biencoder.py