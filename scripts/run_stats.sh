#!/bin/bash

dm_fname=../data/mesh/TNG300-3-Dark/TNG300-3-Dark_dm_33_keep0.10_npix256.h5
npix_orig=256
npix=128

fname_id=TNG
fname_id=catalog.33.threshold1

base_dir=../data/mesh/generated_data
output_dir=${base_dir}/statistics

seed=0
while [ $seed -lt 1 ] 
do
    for stats_name in power cross_power lim_power lim_cross_power
    do
        for sfrmin in 1 10
        do
            python3 calc_stats.py --base_dir $base_dir --output_dir $output_dir --fname_id $fname_id --dm_fname $dm_fname --npix_orig $npix_orig --npix $npix --stats_name $stats_name --seed_start $seed --seed_end $(( seed + 1 )) --sfrmin $sfrmin
        done
    done
    seed=$(( seed + 1 ))
done
