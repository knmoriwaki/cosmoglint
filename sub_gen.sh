#!/bin/bash

boxsize=205 #[Mpc/h]
#npix=436
#npix_z=30

npix=300
npix_z=300

output_dir=./output_test
mkdir -p $output_dir

args="--boxsize $boxsize --npix $npix --npix_z $npix_z"

snapshot_number=38


### group data ###
input_fname=./Transformer/TNG_data/group.${snapshot_number}.txt
output_fname=$output_dir/group.${snapshot_number}.rsd.h5
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_fname --redshift_space

output_fname=$output_dir/group.${snapshot_number}.h5
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_fname 

### subgroup data ###
input_fname=./Transformer/TNG_data/subgroup.${snapshot_number}.txt

output_fname=$output_dir/subgroup.${snapshot_number}.rsd.h5
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_fname --redshift_space

output_fname=$output_dir/subgroup.${snapshot_number}.h5
#python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_fname 

NN_model_dir=./Transformer/NF/output/nl4_bs1024_ep30

for model_name in transformer1 #transformer2 transformer3
do

    input_fname=./Transformer/TNG_data/group.${snapshot_number}.txt
    model_dir=./Transformer/output/${model_name}
    output_fname=$output_dir/group_${model_name}.rsd.h5
    python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_fname --model_dir $model_dir --NN_model_dir $NN_model_dir --redshift_space

    input_fname=./Transformer/TNG_data/group.${snapshot_number}.txt
    model_dir=./Transformer/output/${model_name}
    output_fname=$output_dir/group_${model_name}.h5
    python3 create_data_cube.py $args --input_fname $input_fname --output_fname $output_fname --model_dir $model_dir --NN_model_dir $NN_model_dir

    irun=0
    _args="--boxsize 500 --npix $npix --npix_z $npix_z"

    input_fname=./Pinocchio/my_test/output/pinocchio.2.0000.run${irun}.catalog.out
    model_dir=./Transformer/output/${model_name}
    output_fname=$output_dir/pinocchio.run${irun}.${model_name}.h5
    #python3 create_data_cube.py $_args --input_fname $input_fname --output_fname $output_fname --model_dir $model_dir --NN_model_dir ./Transformer/NN/output 
done

irun=01232
input_fname=./Pinocchio/old_version/output/pinocchio.r${irun}.plc.out
model_dir=./Transformer/output/${model_name}
output_fname=$output_dir/pinocchio.lc${irun}.${model_name}.h5
#python3 create_mock.py $args --input_fname $input_fname --output_fname $output_fname --model_dir $model_dir --NN_model_dir ./Transformer/NN/output 