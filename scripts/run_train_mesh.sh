#!/bin/bash

gpu_id=0

for snapshot_number in 33 #99 #13 15 17 19 21 23 25 29 33 35 40 43 49 55 67 99
do
    config_file=./configs/config_mesh_${snapshot_number}.yaml

    batch_size=256
    for sampler_weight_min in 1 #0.02 #0.05 0.01
    do
        for num_epochs in 500
        do
            output_dir=./runs/output_transformer_mesh/transformer_${snapshot_number}_ep${num_epochs}_bs${batch_size}_w${sampler_weight_min}
            
            mkdir -p $output_dir/source
            cp -r train_transformer.py $config_file ../cosmoglint/model/transformer.py ../cosmoglint/utils $output_dir/source/.

            python3 train_transformer.py --gpu_id $gpu_id --config_file $config_file --output_dir $output_dir --batch_size $batch_size --num_epochs $num_epochs --dropout 0 --sampler_weight_min $sampler_weight_min --save_freq 50 --exclude_ratio 0.5 --sampler_xmin 0.1 --sampler_xmax 0.3
        done
    done
done