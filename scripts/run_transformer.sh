#!/bin/bash

gpu_id=1

snapshot_number=33
data_path=./TNG_data/TNG300-1_${snapshot_number}.h5

num_epochs=20
batch_size=512

num_features_out=100

model_name=transformer1
#model_name=transformer2
#model_name=transformer3

output_dir=./runs/output_transformer/${model_name}_${snapshot_number}_use_vel
mkdir -p $output_dir/source
cp *.py norm_params.txt $output_dir/source/.

python3 train_transformer.py --gpu_id $gpu_id --output_dir $output_dir --data_path $data_path --max_length 50 --num_epochs $num_epochs --batch_size $batch_size --model_name $model_name --num_features_out $num_features_out --dropout 0 --use_sampler --use_dist --use_vel --save_freq 5 
