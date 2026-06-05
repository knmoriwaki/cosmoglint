#!/bin/bash

gpu_id=0

output_dir=../data/mesh/generated_data
mkdir -p $output_dir

snapshot_number=33
run_name=transformer_mesh_${snapshot_number}_ep500_bs256_w1
npix_to_use=128

model_dir=./runs/${run_name}
model_label=_ep300

input_fname=../data/mesh/TNG300-3-Dark/TNG300-3-Dark_dm_33_keep0.10_npix256.h5

num_rounds=8
label=

num_rounds=1
label=.single_round


threshold=1
output_prefix=data_cube.${snapshot_number}.threshold${threshold}.${run_name}${label}
output_catalog_prefix=catalog.${snapshot_number}.threshold${threshold}.${run_name}${label}

seed=0
nseed=100
while [ $seed -lt $nseed ]
do
    output_fname=$output_dir/${output_prefix}.seed${seed}.h5
    output_catalog_fname=$output_dir/${output_catalog_prefix}.seed${seed}.h5

    python3 create_dc_mesh.py --gpu_id $gpu_id --npix_to_use $npix_to_use --npix $npix_to_use --npix_z $npix_to_use --threshold $threshold  --prob_threshold 1e-5 --input_fname $input_fname --output_fname $output_fname --output_catalog_fname $output_catalog_fname --model_dir $model_dir --seed $seed  --num_rounds $num_rounds #--monotonicity_start_index 0
    
    seed=$(( seed + 1 ))
done

for stats_name in power cross_power lim_power lim_cross_power
do
    python3 calc_stats.py --base_dir $output_dir --output_dir $output_dir/statistics --fname_id $output_catalog_prefix --dm_fname $input_fname --stats_name $stats_name --seed_start 0 --seed_end $nseed --sfrmin 1 --npix $npix_to_use
done
