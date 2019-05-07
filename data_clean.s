#!/bin/bash
#BATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=data_clean
#SBATCH --mail-type=END
#SBATCH --mail-user=ssp573@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --cpus-per-task=28

module purge
module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2

source /home/gs3011/pytorch_env/py3.6.3/bin/activate

cd /scratch/gs3011/Yelp-Sentiment-Analysis

python data_clean.py --category Services