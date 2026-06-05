#!/bin/bash

gpu_id=0

threshold=1e-3

base_dir=../data/halo
output_dir=../data/halo/generated_data
mkdir -p $output_dir

snapshot_number=44
snapshot_number_TNG=33
model_name=transformer_global_cond_${snapshot_number}_ep40_bs512_w0.02
model_dir=./runs/${model_name}
#max_sfr_file=../dataset/param_files/max_nbin20_${snapshot_number}.txt

### Create from TNG halo ###
seed=0
while [ $seed -lt 1 ]
do
    global_param_id=0
    while [ $global_param_id -lt 44 ]
    do
        global_param_file=${base_dir}/TNG300-1/params.txt
        
        input_fname=${base_dir}/TNG300-1/TNG300-1_${snapshot_number_TNG}.h5

        o_base=${snapshot_number_TNG}.TNG300-1_test${global_param_id}.threshold${threshold}.${model_name}.seed${seed}
        output_fname=$output_dir/group.data_cube.${o_base}.h5
        output_catalog_fname=$output_dir/group.catalog.${o_base}.h5

        python3 create.py --boxsize 205000 --npix 128 --npix_z 128 --threshold $threshold --gpu_id $gpu_id --prob_threshold 1e-5 --input_fname $input_fname --output_fname $output_fname --output_catalog_fname  $output_catalog_fname --model_dir $model_dir --seed $seed --global_param_file $global_param_file --global_param_id $global_param_id --batch_size 128
    done
    seed=$(( seed + 1 ))
done


### Create from CAMELS halo ###
seed=0
while [ $seed -lt 1 ]
do
    global_param_id=998
    while [ $global_param_id -lt 1000 ]
    do
        global_param_file=${base_dir}/CAMELS/IllustrisTNG/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt
        
        input_fname=${base_dir}/CAMELS/IllustrisTNG/LH/LH_${global_param_id}/my_groups_0${snapshot_number}.hdf5

        o_base=${snapshot_number}.LH${global_param_id}.threshold${threshold}.${model_name}.seed${seed}
        output_fname=$output_dir/group.data_cube.${o_base}.h5
        output_catalog_fname=$output_dir/group.catalog.${o_base}.h5

        #python3 create.py --boxsize 25000 --npix 128 --npix_z 128 --threshold $threshold --gpu_id $gpu_id --prob_threshold 1e-5 --input_fname $input_fname --output_catalog_fname  $output_catalog_fname --model_dir $model_dir --seed $seed --global_param_file $global_param_file --global_param_id $global_param_id
        global_param_id=$(( global_param_id + 1 ))
    done
    seed=$(( seed + 1 ))
done
