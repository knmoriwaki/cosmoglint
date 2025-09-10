#!/bin/bash

gpu_id=1

threshold=1e-3

base_dir=../dataset
output_dir=../dataset/generated_data
mkdir -p $output_dir

snapshot_number=44

model_name=transformer1_with_global_cond
model_dir=./runs/output_transformer/${model_name}_${snapshot_number}_ep40_bs512_w0.02
#max_sfr_file=../dataset/param_files/max_nbin20_${snapshot_number}.txt

seed=0
while [ $seed -lt 1 ]
do
    global_param_id=999
    global_param_file=${base_dir}/CAMELS/IllustrisTNG/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt
    input_fname=${base_dir}/CAMELS/IllustrisTNG/LH/LH_${global_param_id}/my_groups_${snapshot_number}.hdf5

    python3 create_data_cube.py --boxsize 25 --npix 128 --npix_z 128 --threshold $threshold --gpu_id $gpu_id --gen_both --prob_threshold 1e-5 --input_fname $input_fname --output_fname $output_dir/group.data_cube.${snapshot_number}.threshold${threshold}.${model_name}.seed${seed}.h5 --model_dir $model_dir --seed $seed --global_param_file $global_param_file --global_param_id $global_param_id

    seed=$(( seed + 1 ))
done
