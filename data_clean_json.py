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

data=pd.read_json('review.json', lines=True)
reviews = data.drop(['review_id', 'cool', 'funny', 'useful', 'user_id'], 1)
business=pd.read_json('business.json', lines=True)
# import pdb; pdb.set_trace()
# business_filtered=business[~business['categories'].str.contains('Restaurant', na=False) & business['categories'].notnull()]

#df=pd.DataFrame(business_filtered, columns=['categories'])
#bar = df.categories.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
#bar.nlargest(10)

categories = ["Services",
"Home", 
"Shopping",
"Beauty",  
"Services", 
"Health",   
"Spas",    
"Hair",   
"Local"]    

for category in categories:
    print(str(category))
    business_filtered=business[~business['categories'].str.contains('Restaurant', na=False) & business['categories'].str.contains(category, na=False) & business['categories'].notnull()]
#    import pdb; pdb.set_trace()

    new_data=pd.merge(data[['text','stars','business_id']],business_filtered[['business_id','categories']],how='inner',on=['business_id'])
    new_data = data.filter(['text','stars'], axis=1)
    new_data=new_data.sort_values(['stars']).reset_index(drop=True)
    new_data['sentiment'] = np.where(new_data['stars']>=4.0, 'pos', 'neg')
    pos_data=new_data[new_data['sentiment']=='pos']
    neg_data=new_data[new_data['sentiment']=='neg']
    train_data=pd.concat([pos_data[:60000],neg_data[:60000]])
    test_data=pd.concat([pos_data[60000:80000],neg_data[60000:80000]])
    # val_data=pd.concat([pos_data[80000:],neg_data[80000:]])
    print(len(new_data))
    print(len(train_data))

    train_data=train_data.drop('stars',axis=1)
    # val_data=val_data.drop('stars',axis=1)
    test_data=test_data.drop('stars',axis=1)

    # print("Tokenizing val data...")
    # val_data['text']=val_data['text'].apply(tokenize)
    # val_data.to_pickle("data/val_{}_yelp_tokenized_no_stem.pkl".format(categories))
    # print("Val data saved at data/data/val_{}_yelp_tokenized_no_stem.pkl".format(categories))
#    import pdb; pdb.set_trace()
    print("Tokenizing training data...")
    train_data['text']=train_data['text'].apply(tokenize)
    train_data.to_pickle("data/train_{}_yelp_tokenized_no_stem.pkl".format(category))
    print("Train data saved at data/train_{}_yelp_tokenized_no_stem.pkl".format(category))

    print("Tokenizing test data...")
    test_data['text']=test_data['text'].apply(tokenize)
    train_data.to_pickle("data/test_{}_yelp_tokenized_no_stem.pkl".format(category))
    print("Train data saved at data/test_{}_yelp_tokenized_no_stem.pkl".format(category))
