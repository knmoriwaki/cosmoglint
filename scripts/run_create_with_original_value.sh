#!/bin/bash

npix=256 #512
npix_z=256 #512
threshold=1e-3
threshold=1

base_dir=../dataset
output_dir=../dataset/generated_data
mkdir -p $output_dir

snapshot_number=33

sim_name=TNG300-1
boxsize=205 #[Mpc/h]
label=${sim_name}.${snapshot_number}.threshold${threshold}

args="--boxsize $boxsize --npix $npix --npix_z $npix_z --threshold $threshold --gen_both --prob_threshold 1e-5"
args_catalog="$args --gen_catalog --catalog_threshold 1"

input_fname=${base_dir}/${sim_name}/subgroup.${snapshot_number}.txt
#python3 create_data_cube.py $args_catalog --input_fname $input_fname --output_fname $output_dir/subgroup.${label}.cat.txt 

### group data ###
input_fname=${base_dir}/${sim_name}/group.${snapshot_number}.txt
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/group.${label}.h5

### subgroup data ###
input_fname=${base_dir}/${sim_name}/subgroup.${snapshot_number}.txt
python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/subgroup.${label}.h5 

### subgroup data with shuffling ###
input_fname=${base_dir}/${sim_name}/subgroup_shuffled.${snapshot_number}.txt
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/subgroup_shuffled.${label}.h5 

### subgroup at the last corner
input_fname=${base_dir}/${sim_name}/subgroup.${snapshot_number}.txt
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/subgroup.${label}.0.25box.h5 --boxsize_to_use 51.25
