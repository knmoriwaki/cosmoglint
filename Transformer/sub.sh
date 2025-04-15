#!/bin/bash

data_dir=./TNG_data

num_epochs=10
batch_size=512

num_features_out=100

model_name=transformer1
model_name=transformer2
#model_name=transformer3

output_dir=./output/${model_name}
mkdir -p $output_dir/source
cp *.py norm_params.txt $output_dir/source/.

python3 main.py --output_dir $output_dir --data_dir $data_dir --num_epochs $num_epochs --batch_size $batch_size --model_name $model_name --num_features_out $num_features_out --use_sampler #--use_dist --use_loss_mono
