#!/bin/bash

gpu_id=0

threshold=1e-3
threshold=1e-2

side_length=3600 #[arcsec]
fmin=248 #[GHz]
fmax=301 #[GHz]
R=500

### For plotting mean as a function of frequency:
#fmin=100
#fmax=1000
#R=20

output_dir=../dataset/generated_data_lightcone
mkdir -p $output_dir

model_dir=../scripts/runs/output_transformer
input_fname=../dataset/Pinocchio/old_version/output/pinocchio.r01000.plc.out
output_fname=${output_dir}/pinocchio.run${irun}.${side_length}sec_fmin${fmin}_fmax${fmax}_R${R}.h5
python3 create_lightcone.py --input_fname $input_fname --output_fname $output_fname --model_dir $model_dir --threshold $threshold --gpu_id $gpu_id --fmin $fmin --fmax $fmax --R $R --side_length $side_length 
    
    