#!/bin/bash

data_dir=./TNG_data

num_epochs=60
batch_size=1024

num_context=4

model_name=transformer1
#model_name=transformer2
#model_name=transformer3

output_dir=./output/${model_name}
mkdir -p $output_dir/source
cp *.py norm_params.txt $output_dir/source/.

python3 main.py --output_dir $output_dir --data_dir $data_dir --max_length 40 --num_epochs $num_epochs --batch_size $batch_size --model_name $model_name --num_context $num_context --dropout 0.2 --save_freq 20 --use_sampler --use_dist #--load_epoch -1 
