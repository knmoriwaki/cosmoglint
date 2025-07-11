#!/bin/bash

gpu_id=0

threshold=1e-3

side_length=7200 #[arcsec]
angular_resolution=10 # [arcsec]
zmin=0.2
zmax=6.0
dz=0.001


output_dir=../dataset/generated_data_lightcone
mkdir -p $output_dir

model_dir=../scripts/runs/output_transformer

irun=1
irun_id=$(printf "%05d" $irun)
input_fname=../dataset/Pinocchio/my_lightcone/output/pinocchio.r${irun_id}.plc.out

output_fname=${output_dir}/pinocchio.run${irun}.lightcone_sfrd_map.${side_length}sec_zmin${zmin}_zmax${zmax}_dz${dz}_rsd.h5

#python3 create_lightcone.py --input_fname $input_fname --output_fname $output_fname --model_dir $model_dir --threshold $threshold --gpu_id $gpu_id --redshift_min $zmin --redshift_max $zmax --dz $dz --use_logz --side_length $side_length --angular_resolution $angular_resolution --param_dir ../dataset/param_files --redshift_space


catalog_threshold=$threshold
output_fname=${output_dir}/pinocchio.run${irun}.lightcone_catalog.${side_length}sec_zmin${zmin}_zmax${zmax}_sfrmin${catalog_threshold}_rsd.h5

python3 create_lightcone.py --input_fname $input_fname --output_fname $output_fname --model_dir $model_dir --threshold $threshold --gpu_id $gpu_id --redshift_min $zmin --redshift_max $zmax --side_length $side_length --param_dir ../dataset/param_files --gen_catalog --catalog_threshold $catalog_threshold --redshift_space
