#!/bin/bash

gpu_id=1

config_label= # fiducial
#config_label=_33 # training with a single snapshot
#config_label=_mesh_33 # training with mesh condition
#config_label=_global_cond_${snapshot_number} # training with global condition
#config_label=_nf_33 # transformer+nf model

config_file=./configs/config${config_label}.yaml

batch_size=128
for sampler_weight_min in 0.02 #0.05 0.01
do
    for num_epochs in 40 
    do
        output_dir=./runs/transformer${config_label}_bs${batch_size}_w${sampler_weight_min}
        
        mkdir -p $output_dir/source
        cp -r train.py $config_file ../cosmoglint/model/transformer.py ../cosmoglint/utils $output_dir/source/.

        python3 train.py --gpu_id $gpu_id --config_file $config_file --output_dir $output_dir --batch_size $batch_size --num_epochs $num_epochs --dropout 0 --sampler_weight_min $sampler_weight_min --save_freq 10 --exclude_ratio 0.5
    done
done
