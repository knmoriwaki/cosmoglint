#!/bin/bash

gpu_id=0

for snapshot_number in 19 #33 49 67 # 13 15 17 21 23 25 29 35 40 43 55 
do
    data_path="../dataset/TNG300-1/TNG300-1_${snapshot_number}.h5"
    norm_param_file="./param_files/norm_params_${snapshot_number}.txt"

    num_features_out=100
    batch_size=512
    sampler_weight_min=0.02

    model_name=transformer1

    for num_epochs in 1 #60 80
    do
        output_dir=./runs/output_transformer/${model_name}_${snapshot_number}_use_vel_ep${num_epochs}_bs${batch_size}_w${sampler_weight_min}
        mkdir -p $output_dir/source
        cp train_transformer.py $norm_param_file ../cosmoglint/model/transformer.py ../cosmoglint/utils/training_utils.py $output_dir/source/.

        python3 train_transformer.py --gpu_id $gpu_id --output_dir $output_dir --data_path $data_path --use_dist --use_vel --max_length 50 --norm_param_file $norm_param_file --num_epochs $num_epochs --batch_size $batch_size --model_name $model_name --num_features_out $num_features_out --dropout 0 --sampler_weight_min $sampler_weight_min --save_freq 10
    done
done