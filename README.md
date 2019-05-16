# Yelp-Sentiment-Analysis
This project looks at different ways to perform sentiment analysis.

## Steps To Reproduce Experiments
Requirements: Python3.6, pytorch, numpy, torchtext, pandas, matplotlib
* Download fast text vectors "wiki-news-300d-1M.vec.zip" from https://fasttext.cc/docs/en/english-vectors.html
* Unzip and store in a director

### To Run Our Model
Following commands were run on Prince cluster so local or other configurations may require alterations to commands.
* module purge
* module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2
* source <path to virtual env>/pytorch_env/py3.6.3/bin/activate
* cd <path to this repo>
* python main.py --data_dir <path to data> --pretrained_vector_dir <path to vectors> --model <model name>
  
Example:
module purge
module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2

source /home/gs3011/pytorch_env/py3.6.3/bin/activate

cd /scratch/gs3011/Yelp-Sentiment-Analysis

python main.py --data_dir ./data --pretrained_vector_dir /scratch/gs3011/CloudML/wiki-news-300d-1M.vec --model CNN


### To Run The Third Party Models
Note: An API credential will be needed be set up according the appropriate service.

but each script can be run as follows:

python ibm_testing.py
- needs credentials filled into script

python google_testing.py
- needs credentials set in the environment

python amazon_comprehend_testing.py
python clean_amazon_results.py (needed to convert amazon results to just postive and negative scores)
- needs credentials set in the environment
