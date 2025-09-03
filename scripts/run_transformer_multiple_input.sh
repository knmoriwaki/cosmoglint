#!/bin/bash

gpu_id=0

for snapshot_number in 99 #13 15 17 19 21 23 25 29 33 35 40 43 49 55 67 99
do

    data_path="../dataset/TNG300-1/TNG300-1_${snapshot_number}_local_density_1.0Mpc.h5"
    norm_param_file="../dataset/param_files/norm_params.json"

    num_features_out=100
    batch_size=512
    for sampler_weight_min in 0.01 #0.02 #0.05 0.01
    do
        model_name=transformer1

        for num_epochs in 20 40
        do
            output_dir=./runs/output_transformer/${model_name}_${snapshot_number}_ep${num_epochs}_bs${batch_size}_w${sampler_weight_min}_with_local_density
            
            mkdir -p $output_dir/source
            cp train_transformer.py $norm_param_file ../cosmoglint/model/transformer.py ../cosmoglint/utils/io_utils.py $output_dir/source/.

            python3 train_transformer.py --gpu_id $gpu_id --output_dir $output_dir --data_path $data_path --max_length 50 --norm_param_file $norm_param_file --num_epochs $num_epochs --batch_size $batch_size --model_name $model_name --num_features_out $num_features_out --dropout 0 --sampler_weight_min $sampler_weight_min --save_freq 10 --exclude_ratio 0.5 --input_features "HaloMass" "HaloLocalDensity1.0Mpc"
        done
    done
done