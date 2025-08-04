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

side_length=1587.2
angular_resolution=6.2 # [arcsec] Spherex resolution
fmin=130000 #[GHz] -> 2.3um
fmax=300000 #[GHz] -> 1um
R=40
redshift_min=1.1 # Covering Ha and OIII
redshift_max=3.2 # Covering Ha and OIII
        

output_dir=../dataset/generated_data_lightcone
mkdir -p $output_dir

model_dir=../scripts/runs/output_transformer

irun=51
while [ $irun -lt 101 ]
do
    irun_id=$(printf "%05d" $irun)
    input_fname=../dataset/Pinocchio/my_lightcone/output/pinocchio.r${irun_id}.plc.out

    output_fname=${output_dir}/pinocchio.run${irun}.lightcone_intensity_map.${side_length}sec_fmin${fmin}_fmax${fmax}_R${R}_rsd.h5

    #python3 create_mock.py --input_fname $input_fname --output_fname $output_fname --model_dir $model_dir --threshold $threshold --gpu_id $gpu_id --fmin $fmin --fmax $fmax --R $R --side_length $side_length --param_dir ../dataset/param_files --redshift_space

    # For Ha and OIII
    index=0
    while [ $index -lt 100 ]
    do
        output_fname=${output_dir}/pinocchio.run${irun}.index${index}.lightcone_intensity_map.${side_length}sec_fmin${fmin}_fmax${fmax}_R${R}_rsd.h5

        seed=$(( irun * 10000 + index ))
        python3 create_mock.py --input_fname $input_fname --output_fname $output_fname --model_dir $model_dir --threshold $threshold --gpu_id $gpu_id --fmin $fmin --fmax $fmax --R $R --side_length $side_length --angular_resolution $angular_resolution --param_dir ../dataset/param_files --redshift_space --redshift_min $redshift_min --redshift_max $redshift_max --seed $seed --random_patch
        index=$(( index + 1 ))
    done

    
    irun=$(( irun + 1 ))
done    
