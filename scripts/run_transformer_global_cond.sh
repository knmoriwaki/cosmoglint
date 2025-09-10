#!/bin/bash

gpu_id=0

for snapshot_number in 44 #99 #13 15 17 19 21 23 25 29 33 35 40 43 49 55 67 99
do
    base_dir=../dataset
    data_path=${base_dir}/CAMELS/IllustrisTNG/LH/LH_*/my_groups_${snapshot_number}.hdf5
    global_param_file=${base_dir}/CAMELS/IllustrisTNG/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt
    norm_param_file=${base_dir}/param_files/norm_params_CAMELS.json
    indices="0-900"

    num_features_out=100
    batch_size=512
    for sampler_weight_min in 0.02 #0.05 0.01
    do
        model_name=transformer1_with_global_cond

        for num_epochs in 40 
        do
            output_dir=./runs/output_transformer/${model_name}_${snapshot_number}_ep${num_epochs}_bs${batch_size}_w${sampler_weight_min}
            
            mkdir -p $output_dir/source
            cp train_transformer.py $norm_param_file ../cosmoglint/model/transformer.py ../cosmoglint/utils/io_utils.py $output_dir/source/.

            python3 train_transformer.py --gpu_id $gpu_id --output_dir $output_dir --data_path "$data_path" --indices $indices --max_length 50 --global_param_file $global_param_file --norm_param_file $norm_param_file --num_epochs $num_epochs --batch_size $batch_size --model_name $model_name --num_features_out $num_features_out --dropout 0 --sampler_weight_min $sampler_weight_min --save_freq 10 --global_features Omega_m sigma_8 A_SN1 A_AGN1 A_SN2 A_AGN2

        done
    done
done