import pandas as pd
import numpy as np
import spacy
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

porter_stemmer=PorterStemmer()

tokenizer = spacy.load('en_core_web_sm')
punctuations = list(string.punctuation)

def tokenize(sent):
  tokens = tokenizer(sent)
  unwanted=set(list(punctuations)+['<br','/><br','\n','\t'])
  return [token.text.lower() for token in tokens if (token.text not in unwanted)]

data=pd.read_csv('review.csv')
business=pd.read_csv('business.csv')
business_filtered=business[business['categories'].str.contains('Restaurant') & business['categories'].notnull()]
new_data=pd.merge(data[['text','stars','business_id']],business_filtered[['business_id','categories']],how='inner',on=['business_id'])
new_data = data.filter(['text','stars'], axis=1)
new_data=new_data.sort_values(['stars']).reset_index(drop=True)
new_data['sentiment'] = np.where(new_data['stars']>=4.0, 'pos', 'neg')
pos_data=filtered_data[filtered_data['sentiment']=='pos']
neg_data=filtered_data[filtered_data['sentiment']=='neg']
train_data=pd.concat([pos_data[:60000],neg_data[:60000]])
test_data=pd.concat([pos_data[60000:80000],neg_data[60000:80000]])
val_data=pd.concat([pos_data[80000:],neg_data[80000:]])

train_data=train_data.drop('stars',axis=1)
val_data=val_data.drop('stars',axis=1)
test_data=test_data.drop('stars',axis=1)

print("Tokenizing val data...")
val_data['text']=val_data['text'].apply(tokenize)
val_data.to_pickle("data/val_restaurants_tokenized_no_stem.pkl")
print("Val data saved at data/val_restaurants_tokenized_no_stem.pkl")

print("Tokenizing training data...")
train_data['text']=train_data['text'].apply(tokenize)
train_data.to_pickle("data/train_restaurants_tokenized_no_stem.pkl")
print("Training data saved at data/train_restaurants_tokenized_no_stem.pkl")

print("Tokenizing test data...")
test_data['text']=test_data['text'].apply(tokenize)
test_data.to_pickle("data/test_restaurants_tokenized.pkl")
print("Training data saved at data/test_restaurants_tokenized_no_stem.pkl")
