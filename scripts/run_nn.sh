#!/bin/bash

data_dir=../TNG_data

num_epochs=10
batch_size=1024

output_dir=./runs/output_nn
mkdir -p $output_dir/source
cp *.py ../norm_params.txt $output_dir/source/.

python3 train_nn.py --output_dir $output_dir --data_dir $data_dir --num_epochs $num_epochs --batch_size $batch_size --use_sampler 
