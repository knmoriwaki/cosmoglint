#!/bin/bash

boxsize=205 #[Mpc/h]
#npix=436
#npix_z=30

npix=256
npix_z=256

output_dir=./output_test
mkdir -p $output_dir

args="--boxsize $boxsize --npix $npix --npix_z $npix_z"

snapshot_number=33

### group data ###
input_fname=./Transformer/TNG_data/group.${snapshot_number}.txt
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/group.${snapshot_number}.h5 --gen_both
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/group.cat.${snapshot_number}.txt --gen_catalog --catalog_threshold 10 --gen_both

### subgroup data ###
input_fname=./Transformer/TNG_data/subgroup.${snapshot_number}.txt
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/subgroup.${snapshot_number}.h5 --gen_both 
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname output_fname=$output_dir/subgroup.cat.${snapshot_number}.txt --gen_catalog --catalog_threshold 10 --gen_both


for model_name in transformer1 #transformer2 transformer3
do
    model_dir=./Transformer/output/${model_name}_${snapshot_number}_use_vel

    input_fname=./Transformer/TNG_data/group.${snapshot_number}.txt
    python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/group.${snapshot_number}.${model_name}.h5 --model_dir $model_dir --gen_both
    #python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/group.cat.${snapshot_number}.${model_name}.txt --model_dir $model_dir --redshift_space --gen_catalog --catalog_threshold 10
    #python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_dir/group.cat.${snapshot_number}.${model_name}.txt --model_dir $model_dir --gen_catalog --catalog_threshold 10

    irun=3
    _args="--boxsize 800 --npix 1000 --npix_z 1000"
    input_fname=./Pinocchio/my_test/output/pinocchio.2.0000.run${irun}.catalog.out    
    python3 create_data_cube.py $_args --input_fname $input_fname --output_fname $output_dir/pinocchio.run${irun}.${model_name}.h5 --model_dir $model_dir --gen_both
    python3 create_data_cube.py $_args --input_fname $input_fname --output_fname $output_dir/pinocchio.run${irun}.${model_name}.mass_corrected.h5 --model_dir $model_dir --mass_correction_factor 0.8 --gen_both
    
done
