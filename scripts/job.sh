#!/bin/bash

TIME=$(date +"%Y%m%d_%H%M%S")
nohup ./run_transformer.sh > ./tmp/out_${TIME}.log 2>&1 &
#nohup ./run_transformer_nf.sh > ./tmp/out_${TIME}.log 2>&1 &
