import pandas as pd
import numpy as np
import argparse
from dataloader import * 
import boto3
import spacy
import string
import csv

tokenizer = spacy.load('en_core_web_sm')
punctuations = list(string.punctuation)

def tokenize(sent):
  tokens = tokenizer(sent)
  unwanted=set(list(punctuations)+['<br','/><br','\n','\t'])
  return [token.text.lower() for token in tokens if (token.text not in unwanted)]

client = boto3.client('comprehend')

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--data_dir', metavar='N', type=str,
                    help='Data directory',default='data')
args = parser.parse_args()

# test_tokenized= pd.read_pickle(args.data_dir+'/test_restaurants_tokenized_no_stem.pkl')#[:2000]

# sample = test_tokenized.sample(frac=0.1, random_state=1)

# print("Tokenizing test data...")
# sample.to_pickle("data/comparison_test_set.pkl")
# print("Training data saved at data/comparison_test_set.pkl")

test_tokenized= pd.read_pickle(args.data_dir+'/comparison_test_set.pkl')[:1000]

batch = []
labels = []
num_correct = 0
total_values = len(test_tokenized)
raw_results = [] # each item is {
        #     'Index' : item['Index'],
        #     'Label' : item['Label'],
        #     'Prediction' : item['Prediction'],
        #     'Positive%' : item['Positive%'],
        #     'Negative%' : item['Negative%'],
        #     'Neutral%' : item['Neutral%'],
        #     'Mixed%' : item['Mixed%']
        # }

for text, label in test_tokenized.values.tolist():
    if len(batch) == 25:
        results = client.batch_detect_sentiment(TextList=batch, LanguageCode='en')
        num_correct_in_batch = len(np.where(np.array([x['Sentiment'] for x in results['ResultList']])==labels)[0])
        num_correct += num_correct_in_batch
        
        for item in results['ResultList']:
            raw_results.append({
            'Index' : item['Index'],
            'Label' : labels[int(item['Index'])],
            'Prediction' : item['Sentiment'],
            'Positive%' : item['SentimentScore']['Positive'],
            'Negative%' : item['SentimentScore']['Negative'],
            'Neutral%' : item['SentimentScore']['Neutral'],
            'Mixed%' : item['SentimentScore']['Mixed']
            })
        batch = []
        labels = []

    batch.append(" ".join(text))
    labels.append("POSITIVE" if label == 'pos' else "NEGATIVE")

print("Amazon Comprehend Predicted {} out of {} or {} correctly".format(num_correct, total_values, str(num_correct / total_values) + '%'))

file = open("./plots/AmazonComprehendTestData.csv", 'w')
with file:
    fnames = ['Index', 'Label', 'Prediction', 'Positive%', 'Negative%', 'Neutral%', 'Mixed%']
    writer = csv.DictWriter(file, fieldnames=fnames)  
    writer.writeheader()
    for item in raw_results:
        writer.writerow({
            'Index' : item['Index'],
            'Label' : item['Label'],
            'Prediction' : item['Prediction'],
            'Positive%' : item['Positive%'],
            'Negative%' : item['Negative%'],
            'Neutral%' : item['Neutral%'],
            'Mixed%' : item['Mixed%']
        })