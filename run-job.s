#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:k80:4
#SBATCH --mem=250GB
#SBATCH --time=1:00:00
#SBATCH --job-name=sent
#SBATCH --output=sent_%j.out
#SBATCH --exclusive
#SBATCH --reservation=chung
#SBATCH --cpus-per-task=28

module purge
module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2

source /home/gs3011/pytorch_env/py3.6.3/bin/activate

cd /scratch/gs3011/Yelp-Sentiment-Analysis

python main.py
