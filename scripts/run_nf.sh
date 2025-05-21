#!/bin/bash

data_dir=./TNG_data

num_layers=4
batch_size=1024
num_epochs=30

output_dir=./runs/output_nf/nl${num_layers}_bs${batch_size}_ep${num_epochs}
mkdir -p $output_dir/source
cp *.py ../norm_params.txt $output_dir/source/.

python3 train_nf.py --output_dir $output_dir --data_dir $data_dir --num_epochs $num_epochs --batch_size $batch_size --num_layers $num_layers --lr 5e-4 --use_sampler 
