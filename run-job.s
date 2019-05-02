
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:4
#SBATCH --mem=150GB
#SBATCH --time=5:00:00
#SBATCH --job-name=sentiment_analysis
#SBATCH --output=sentiment_analysis_%j.out
##SBATCH --exclusive
##SBATCH --reservation=chung
#SBATCH --cpus-per-task=28

module purge
module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2

source /home/gs3011/pytorch_env/py3.6.3/bin/activate

cd /scratch/gs3011/Yelp-Sentiment-Analysis

python main.py --data_dir ./data --pretrained_vector_dir /scratch/gs3011/CloudML/wiki-news-300d-1M.vec --model CNN

