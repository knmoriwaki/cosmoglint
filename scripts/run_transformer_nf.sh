#!/bin/bash

gpu_id=1

snapshot_number=33
data_path=../dataset/TNG300-1/TNG300-1_${snapshot_number}.h5

num_epochs=1
batch_size=512

model_name=transformer1
#model_name=transformer2
#model_name=transformer3

base_dist=conditional_normal
num_context=16
num_layers=6
num_flows=8

lr=1e-4

norm_param_file=../dataset/param_files/norm_params_${snapshot_number}.txt

output_dir=./runs/output_transformer_nf/${model_name}_${snapshot_number}_${base_dist}_nc${num_context}_nl${num_layers}_nf${num_flows}_ep${num_epochs}_lr${lr}_use_vel
mkdir -p $output_dir/source
cp train_transformer_nf.py ${norm_param_file} ../cosmoglint/model/transformer_nf.py ../cosmoglint/utils/training_utils.py $output_dir/source/.

python3 train_transformer_nf.py --gpu_id $gpu_id --output_dir $output_dir --data_path $data_path --max_length 40 --num_epochs $num_epochs --batch_size $batch_size --model_name $model_name --base_dist $base_dist --num_context $num_context --num_layers $num_layers --num_flows $num_flows --lr $lr --dropout 0 --save_freq 20 --use_dist --use_vel --use_sampler --norm_param_file $norm_param_file
