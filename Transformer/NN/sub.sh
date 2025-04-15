#!/bin/bash

data_dir=../TNG_data

num_epochs=200
batch_size=1024

num_features_out=100

output_dir=./output
mkdir -p $output_dir/source
cp *.py ../norm_params.txt $output_dir/source/.

python3 main.py --output_dir $output_dir --data_dir $data_dir --num_epochs $num_epochs --batch_size $batch_size --use_sampler 
