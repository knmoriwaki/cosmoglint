#!/bin/bash

gpu_id=1

threshold=1e-3
#threshold=1

base_dir=../dataset
output_dir=../dataset/generated_data
mkdir -p $output_dir

snapshot_number=33

model_name=transformer1
model_dir=./runs/output_transformer/${model_name}_${snapshot_number}_ep40_bs512_w0.02
max_sfr_file=../dataset/param_files/max_nbin20_${snapshot_number}.txt

seed=0
while [ $seed -lt 1 ]
do
    input_fname=${base_dir}/TNG300-1/group.${snapshot_number}.txt
    #python3 create_data_cube.py --boxsize 205 --npix 512 --npix_z 512 --threshold $threshold --gpu_id $gpu_id --gen_both --prob_threshold 1e-5 --input_fname $input_fname --output_fname $output_dir/group.data_cube.${snapshot_number}.threshold${threshold}.${model_name}.seed${seed}.h5 --model_dir $model_dir --max_sfr_file $max_sfr_file --seed $seed
    #python3 create_data_cube.py --boxsize 205 --npix 512 --npix_z 512 --threshold 1e-1 --gpu_id $gpu_id --gen_both --prob_threshold 1e-5 --input_fname $input_fname --output_fname $output_dir/group.catalog.${snapshot_number}.threshold${threshold}.${model_name}.seed${seed}.txt --model_dir $model_dir --max_sfr_file $max_sfr_file --seed $seed --gen_catalog --catalog_threshold 1e-1

    input_fname=${base_dir}/TNG300-1-Dark/group.${snapshot_number}.txt
    mass_correction_factor=0.9
    python3 create_data_cube.py --boxsize 205 --npix 512 --npix_z 512 --threshold $threshold --gpu_id $gpu_id --gen_both --prob_threshold 1e-5 --input_fname $input_fname --output_fname $output_dir/group_dm_only.data_cube.${snapshot_number}.threshold${threshold}.${model_name}.seed${seed}.factor${mass_correction_factor}.h5 --model_dir $model_dir --max_sfr_file $max_sfr_file --seed $seed --mass_correction_factor $mass_correction_factor
    #python3 create_data_cube.py --boxsize 205 --npix 512 --npix_z 512 --threshold 1e-1 --gpu_id $gpu_id --gen_both --prob_threshold 1e-5 --input_fname $input_fname --output_fname $output_dir/group_dm_only.catalog.${snapshot_number}.threshold${threshold}.${model_name}.seed${seed}.txt --model_dir $model_dir --max_sfr_file $max_sfr_file --seed $seed --gen_catalog --catalog_threshold 1e-1

    ### Piniocchio 
    args_pin="--boxsize 677.4 --npix 1692 --npix_z 1692 --threshold $threshold --gpu_id $gpu_id --gen_both"
    label_pin=threshold${threshold}

    irun=2
    input_fname=${base_dir}/Pinocchio/my_test/output/pinocchio.2.0000.run${irun}.catalog.out    
    python3 create_data_cube.py $args_pin --input_fname $input_fname --output_fname $output_dir/pinocchio.run${irun}.data_cube.${snapshot_number}.${label_pin}.${model_name}.factor${mass_correction_factor}.h5 --model_dir $model_dir --max_sfr_file $max_sfr_file --mass_correction_factor $mass_correction_factor
    
    seed=$(( seed + 1 ))
done
