#!/bin/bash
#BATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:k80:1
#SBATCH --time=40:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=vi_cnn_encoder_512_5
#SBATCH --mail-type=END
#SBATCH --mail-user=ssp573@nyu.edu
#SBATCH --output=slurm_%j.out

#module purge

SRCDIR=$HOME
RUNDIR=$SCRATCH/CloudML/Proj_3/
mkdir -p $RUNDIR

cd $SLURM_SUBMIT_DIR
$SRCDIR
cp -r $SRCDIR/CloudML/Yelp-Sentiment-Analysis $RUNDIR

cd $RUNDIR

source ~/pytorch_env/py3.6.3/bin/activate

pwd
python3 ./Yelp-Sentiment-Analysis/main.py --data_dir ./Yelp-Sentiment-Analysis/data
