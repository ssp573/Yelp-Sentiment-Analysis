import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
spacy_en=spacy.load('en')
import spacy
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import torchtext.data as data
import pandas as pd
from collections import Counter
import numpy as np



train_tokenized=pd.read_pickle('data/train_restaurants_tokenized.pkl')
val_tokenized=pd.read_pickle('data/val_restaurants_tokenized.pkl')
test_tokenized= pd.read_pickle('data/test_restaurants_tokenized.pkl')

train_tokenized['sentiment']=np.where(train_tokenized['sentiment']=='pos',1,0)
val_tokenized['sentiment']=np.where(val_tokenized['sentiment']=='pos',1,0)
test_tokenized['sentiment']=np.where(test_tokenized['sentiment']=='pos',1,0)

train_tuples= [tuple(x) for x in train_tokenized[['text','sentiment']].values]
train_data_tokens,train_data_labels=zip(*train_tuples)

val_tuples= [tuple(x) for x in val_tokenized[['text','sentiment']].values]
val_data_tokens,val_data_labels=zip(*val_tuples)

test_tuples= [tuple(x) for x in test_tokenized[['text','sentiment']].values]
test_data_tokens,test_data_labels=zip(*test_tuples)

tokens=list(train_tokenized['text'])
all_tokens=[token for token_list in tokens for token in token_list]
dist_tokens=list(set(all_tokens))

max_vocab_size=10000
PAD_IDX=0
UNK_IDX=1

def build_vocab(all_tokens):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

token2id, id2token = build_vocab(dist_tokens)

def token2index_dataset(tokens_data):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data

train_data_indices = token2index_dataset(train_data_tokens)
val_data_indices = token2index_dataset(val_data_tokens)
test_data_indices = token2index_dataset(test_data_tokens)

#Dataloader randomly selects batch of data from the dataset

MAX_SENTENCE_LENGTH = 200

import numpy as np
import torch
from torch.utils.data import Dataset

class YelpDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self,  data_list, target_list):
        """
        @param data_list: list of newsgroup tokens 
        @param target_list: list of newsgroup targets 

        """
        self.data_list = data_list
        self.target_list = target_list
        #print(len(self.data_list),len(self.target_list))
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        
        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]

def yelp_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[1])), 
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    #print(label_list[:10])
    return [torch.from_numpy(np.array(data_list)).long(), torch.LongTensor(length_list), torch.LongTensor(label_list)]

BATCH_SIZE = 16
train_dataset = YelpDataset(train_data_indices, train_data_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=yelp_collate_func,
                                           shuffle=True)

val_dataset = YelpDataset(val_data_indices, val_data_labels)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=yelp_collate_func,
                                           shuffle=True)

test_dataset = YelpDataset(test_data_indices, test_data_labels)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=yelp_collate_func,
                                           shuffle=False)



hidden_size=512

class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, emb_dim):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        # pay attention to padding_idx 
        #embed dimension should be atleast 100
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim,hidden_size)
        self.linear2=nn.Linear(hidden_size,2)
    
    def forward(self, data, length):
        """
        
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        #data.type()
        #print(data.type())
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()
     
        # return logits
        out = F.relu(self.linear(out.float()))
        out=self.linear2(out)
        return out

emb_dim = 100
print(len(id2token))
model = BagOfWords(len(id2token), emb_dim)


emb_dim = 100
#print(len(id2token))
model = BagOfWords(len(id2token), emb_dim)

learning_rate = 0.001
num_epochs = 10 # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#optimizer= torch.optim.SGD(model.parameters(), lr=learning_rate)

# Function for testing the model
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = data, lengths, labels
        #print(data_batch,label_batch,length_batch)
        outputs = F.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)

val_acc_list=[]
train_acc_list=[]
for epoch in range(num_epochs):
    #linear annealing of learning rate at every 4th epoch
    if epoch%3==2:
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate*0.5)
    for i, (data, lengths, labels) in enumerate(train_loader):
        model.train()
        data_batch, length_batch, label_batch = data, lengths, labels
        optimizer.zero_grad()
        #for k in label_batch:
        #    print(k.type())
        outputs = model(data_batch, length_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()
        # validate every 100 iterations
        if i > 0 and i % 100 == 0:
            # validate
            val_acc = test_model(val_loader, model)
            train_acc= test_model(train_loader, model)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}, Training Acc: {}'.format( 
                       epoch+1, num_epochs, i+1, len(train_loader), val_acc, train_acc))
    val_acc_list.append(val_acc)
    train_acc_list.append(train_acc)
    
#import matplotlib.pyplot as plt
#%matplotlib inline

#epochs=[i for i in range(1,num_epochs+1)]

#plt.subplot(223)
#plt.plot( epochs,train_acc_list, label="training accuracy")
#plt.plot( epochs, val_acc_list, label="validation accuracy")
#plt.ylabel("accuracy")
#plt.xlabel("number of epochs")
# Place a legend to the right of this smaller subplot.
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#plt.show()


