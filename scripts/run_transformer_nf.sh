#!/bin/bash

gpu_id=1

snapshot_number=33
data_path=../dataset/TNG300-1/TNG300-1_${snapshot_number}.h5

num_epochs=300
batch_size=512

model_name=transformer1
#model_name=transformer2
#model_name=transformer3

#base_dist=conditional_normal
base_dist=normal
num_context=16
num_layers=6
num_flows=8

lr=1e-4
#lr=5e-5
#lr=1e-5

norm_param_file=../dataset/param_files/norm_params_${snapshot_number}.json

output_dir=./runs/output_transformer_nf/${model_name}_${snapshot_number}_${base_dist}_nc${num_context}_nl${num_layers}_nf${num_flows}_ep${num_epochs}_lr${lr}
mkdir -p $output_dir/source
cp train_transformer_nf.py ${norm_param_file} ../cosmoglint/model/transformer_nf.py ../cosmoglint/utils/io_utils.py $output_dir/source/.

python3 train_transformer_nf.py --gpu_id $gpu_id --output_dir $output_dir --data_path $data_path --max_length 50 --num_epochs $num_epochs --batch_size $batch_size --model_name $model_name --base_dist $base_dist --num_context $num_context --num_layers $num_layers --num_flows $num_flows --lr $lr --dropout 0 --save_freq 20 --sampler_weight_min 0.02 --norm_param_file $norm_param_file --exclude_ratio 0.5
