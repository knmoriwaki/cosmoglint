#!/bin/bash

gpu_id=0

npix=512
npix_z=512
threshold=1e-3

base_dir=../dataset
output_dir=../dataset/generated_data
mkdir -p $output_dir

snapshot_number=33

sim_name=TNG300-1

### Arguments for TNG
boxsize=205 #[Mpc/h]
label=${sim_name}.${snapshot_number}.threshold${threshold}

args="--boxsize $boxsize --npix $npix --npix_z $npix_z --threshold $threshold --gpu_id $gpu_id --gen_both --prob_threshold 2e-5"
args_catalog="$args --gen_catalog --catalog_threshold 10"

### Arguments for piniocchio 
args_pin="--boxsize 677.4 --npix 846 --npix_z 846 --threshold $threshold --gpu_id $gpu_id --gen_both"
label_pin=threshold${threshold}.low_res

args_pin="--boxsize 677.4 --npix 1692 --npix_z 1692 --threshold $threshold --gpu_id $gpu_id --gen_both"
label_pin=threshold${threshold}
args_pin_catalog="$args_pin --gen_catalog --catalog_threshold 10"


### group data ###
input_fname=${base_dir}/${sim_name}/group.${snapshot_number}.txt
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/group.${label}.h5

### subgroup data ###
input_fname=${base_dir}/${sim_name}/subgroup.${snapshot_number}.txt
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/subgroup.${label}.h5 
#python3 create_data_cube.py $args_catalog --input_fname $input_fname --output_fname output_fname=$output_dir/subgroup.${snapshot_number}.cat.txt 

model_name=transformer1
model_dir=./runs/output_transformer/${model_name}_${snapshot_number}_use_vel_ep40_bs512_w0.02
max_ids_file=./param_files/max_ids_20_${snapshot_number}.txt

seed=0
while [ $seed -lt 100 ]
do
    input_fname=${base_dir}/${sim_name}/group.${snapshot_number}.txt
    python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/group.${label}.${model_name}.seed${seed}.h5 --model_dir $model_dir --max_ids_file $max_ids_file --seed $seed
    #python3 create_data_cube.py $args_catalog --input_fname $input_fname --output_fname $output_dir/group.${label}.${model_name}.cat.txt --model_dir $model_dir 

    irun=2
    input_fname=${base_dir}/Pinocchio/my_test/output/pinocchio.2.0000.run${irun}.catalog.out    
    #python3 create_data_cube.py $args_pin --input_fname $input_fname --output_fname $output_dir/pinocchio.run${irun}.${label_pin}.${model_name}.h5 --model_dir $model_dir --max_ids_file $max_ids_file
    #python3 create_data_cube.py $args_pin_catalog --input_fname $input_fname --output_fname $output_dir/pinocchio.run${irun}.${label_pin}.${model_name}.mass_corrected.h5 --model_dir $model_dir --mass_correction_factor 0.8 

    seed=$(( seed + 1 ))
done
