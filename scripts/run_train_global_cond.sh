#!/bin/bash

gpu_id=0

for snapshot_number in 44 
do
    config_file=./configs/config_global_cond_${snapshot_number}.yaml

    batch_size=512
    for sampler_weight_min in 0.02 #0.05 0.01
    do
        for num_epochs in 40 
        do
            output_dir=./runs/output_transformer/$transformer_${snapshot_number}_ep${num_epochs}_bs${batch_size}_w${sampler_weight_min}
            
            mkdir -p $output_dir/source
            cp -r train_transformer.py $config_file ../cosmoglint/model/transformer.py ../cosmoglint/utils $output_dir/source/.

            python3 train_transformer.py --gpu_id $gpu_id --config_file $config_file --output_dir $output_dir --batch_size $batch_size --num_epochs $num_epochs --dropout 0 --sampler_weight_min $sampler_weight_min --save_freq 10

        done
    done
done