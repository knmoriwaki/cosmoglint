#!/bin/bash

gpu_id=0

threshold=1e-3

#base_dir=../dataset
#base_dir=/mnt/data/moriwaki/cosmoglint/dataset
base_dir=/mnt/data/moriwaki/cosmoglint/data/halo
#output_dir=../dataset/generated_data
output_dir=/mnt/data/ito/cosmoglint/scripts/runs/output_transformer/catalog_all
mkdir -p $output_dir

snapshot_number=44

model_name=transformer1_with_global_cond
model_dir=./runs/output_transformer/${model_name}_${snapshot_number}_ep40_bs512_w0.02
#max_sfr_file=../dataset/param_files/max_nbin20_${snapshot_number}.txt

seed=0
while [ $seed -lt 1 ]
do
    for global_param_id in {900..999}
    do
        global_param_file=${base_dir}/CAMELS/IllustrisTNG/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt
        input_fname=${base_dir}/CAMELS/IllustrisTNG/LH/LH_${global_param_id}/my_groups_${snapshot_number}.hdf5

        #output_fname=$output_dir/group_${global_param_id}.data_cube.${snapshot_number}.threshold${threshold}.${model_name}.seed${seed}.h5
        #python3 create_data_cube.py --boxsize 25 --npix 128 --npix_z 128 --threshold $threshold --gpu_id $gpu_id --gen_both --prob_threshold 1e-5 --input_fname $input_fname --output_fname $output_fname --model_dir $model_dir --seed $seed --global_param_file $global_param_file --global_param_id $global_param_id
        
        output_fname=$output_dir/group_${global_param_id}.data_cube.${snapshot_number}.threshold${threshold}.${model_name}.seed${seed}.h5 
        output_catalog_fname=$output_dir/group_${global_param_id}.catalog_data.${snapshot_number}.threshold${threshold}.${model_name}.seed${seed}.h5 
        python3 create_data_cube.py --boxsize 25 --npix 128 --npix_z 128 --threshold $threshold --catalog_threshold 0.1 --gpu_id $gpu_id --gen_both --prob_threshold 1e-5 --input_fname $input_fname --output_catalog_fname $output_catalog_fname --output_fname $output_fname --model_dir $model_dir --seed $seed --global_param_file $global_param_file --global_param_id $global_param_id

    done
    seed=$(( seed + 1 ))
done