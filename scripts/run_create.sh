#!/bin/bash

gpu_id=1

boxsize=205 #[Mpc/h]
#npix=436
#npix_z=30

npix=256
npix_z=256
threshold=1e-3

output_dir=./generated_data
mkdir -p $output_dir

snapshot_number=33

### Arguments for TNG
args="--boxsize $boxsize --npix $npix --npix_z $npix_z --threshold $threshold --gpu_id $gpu_id --gen_both"
args_catalog="$args --gen_catalog --catalog_threshold 10"

### Arguments for piniocchio 
args_pin="--boxsize 677.4 --npix 846 --npix_z 846 --threshold $threshold --gpu_id $gpu_id --gen_both"
args_pin_catalog="$args_pin --gen_catalog --catalog_threshold 10"


### group data ###
input_fname=./TNG_data/group.${snapshot_number}.txt
python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/group.${snapshot_number}.h5

### subgroup data ###
input_fname=./TNG_data/subgroup.${snapshot_number}.txt
python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/subgroup.${snapshot_number}.h5 
#python3 create_data_cube.py $args_catalog --input_fname $input_fname --output_fname output_fname=$output_dir/subgroup.${snapshot_number}.cat.txt 


for model_name in transformer1 #transformer2 transformer3
do
    model_dir=./runs/output_transformer/${model_name}_${snapshot_number}_use_vel

    input_fname=./TNG_data/group.${snapshot_number}.txt
    python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/group.${snapshot_number}.${model_name}.h5 --model_dir $model_dir
    #python3 create_data_cube.py $args_catalog --input_fname $input_fname --output_fname $output_dir/group.${snapshot_number}.${model_name}.cat.txt --model_dir $model_dir 

    irun=2
    
    input_fname=./Pinocchio/my_test/output/pinocchio.2.0000.run${irun}.catalog.out    
    python3 create_data_cube.py $args_pin --input_fname $input_fname --output_fname $output_dir/pinocchio.run${irun}.${model_name}.h5 --model_dir $model_dir 
    python3 create_data_cube.py $args_pin --input_fname $input_fname --output_fname $output_dir/pinocchio.run${irun}.${model_name}.mass_corrected.h5 --model_dir $model_dir --mass_correction_factor 0.8 

    #python3 create_data_cube.py $args_pin_catalog --input_fname $input_fname --output_fname $output_dir/pinocchio.run${irun}.${model_name}.mass_corrected.h5 --model_dir $model_dir --mass_correction_factor 0.8 
    
done
