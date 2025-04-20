#!/bin/bash

gpu_id=1

data_dir=./TNG_data

num_epochs=200
batch_size=1024

lr=2e-4

base_dist=conditional_gaussian
num_context=16
num_layers=6
num_flows=6

model_name=transformer1
#model_name=transformer2
#model_name=transformer3

output_dir=./output/${model_name}_${base_dist}_nc${num_context}_nl${num_layers}_nf${num_flows}_ep${num_epochs}_lr${lr}
mkdir -p $output_dir/source
cp *.py norm_params.txt $output_dir/source/.

python3 main.py --gpu_id $gpu_id --output_dir $output_dir --data_dir $data_dir --max_length 40 --num_epochs $num_epochs --batch_size $batch_size --model_name $model_name --num_context $num_context --num_layers $num_layers --num_flows $num_flows --lr $lr --dropout 0.2 --save_freq 20 --use_sampler --use_dist --use_vel #--load_epoch -1 
