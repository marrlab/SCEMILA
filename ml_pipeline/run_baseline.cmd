#!/bin/bash

#SBATCH -o ./out.txt

#SBATCH -e ./error.txt

#SBATCH -J stability_amil

#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH -x icb-gpusrv[01-02]

#SBATCH -c 4

#SBATCH --mem-per-cpu 2000

#SBATCH --gres=gpu:1

#SBATCH -t 48:00:00

#SBATCH --nice=10000

source ~/anaconda/etc/profile.d/conda.sh
conda activate pytorch_latest 

python3 run_pipeline.py --result_folder pub_singleatt_0 --fold 0 --filter_mediocre_quality=0 --filter_diff=20 --save_model=1 --multi_att=0
python3 run_pipeline.py --result_folder pub_singleatt_1 --fold 1 --filter_mediocre_quality=0 --filter_diff=20 --save_model=1 --multi_att=0
python3 run_pipeline.py --result_folder pub_singleatt_2 --fold 2 --filter_mediocre_quality=0 --filter_diff=20 --save_model=1 --multi_att=0
python3 run_pipeline.py --result_folder pub_singleatt_3 --fold 3 --filter_mediocre_quality=0 --filter_diff=20 --save_model=1 --multi_att=0
python3 run_pipeline.py --result_folder pub_singleatt_4 --fold 4 --filter_mediocre_quality=0 --filter_diff=20 --save_model=1 --multi_att=0

