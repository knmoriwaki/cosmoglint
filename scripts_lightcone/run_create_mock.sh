#!/bin/bash

gpu_id=0

threshold=1e-3

side_length=7200 #[arcsec]
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

irun=2
while [ $irun -lt 51 ]
do
    irun_id=$(printf "%05d" $irun)
    input_fname=../dataset/Pinocchio/my_lightcone/output/pinocchio.r${irun_id}.plc.out

    output_fname=${output_dir}/pinocchio.run${irun}.${side_length}sec_fmin${fmin}_fmax${fmax}_R${R}.h5

    python3 create_mock.py --input_fname $input_fname --output_fname $output_fname --model_dir $model_dir --threshold $threshold --gpu_id $gpu_id --fmin $fmin --fmax $fmax --R $R --side_length $side_length --param_dir ../dataset/param_files
    
    irun=$(( irun + 1 ))
done    