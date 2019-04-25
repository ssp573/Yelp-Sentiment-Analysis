import torchtext.data as data
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from dataloader import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--data_dir', metavar='N', type=str,
                    help='Data directory',default='data')
parser.add_argument('--hidden_size', metavar='N', type=int,
                    help='Hidden layer size',default=512)
parser.add_argument('--emb_dim', metavar='N', type=int,
                    help='Dimensions for word embedding',default=512)
parser.add_argument('--max_vocab_size', metavar='N', type=int,
                    help='Maximum vocabulary size',default=10000)
parser.add_argument('--batch_size', metavar='N', type=int,
                    help='Batch size',default=16)
parser.add_argument('--model',metavar='N', type=str,
                    help='Model',default='BOW')
parser.add_argument('--optimizer',metavar='N', type=str,
                    help='Optimizer',default='adam')
args = parser.parse_args()



train_tokenized=pd.read_pickle(args.data_dir+'/train_restaurants_tokenized.pkl')
val_tokenized=pd.read_pickle(args.data_dir+'/val_restaurants_tokenized.pkl')
test_tokenized= pd.read_pickle(args.data_dir+'/test_restaurants_tokenized.pkl')

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

token2id, id2token = build_vocab(dist_tokens,args.max_vocab_size)

train_data_indices = token2index_dataset(train_data_tokens,token2id)
val_data_indices = token2index_dataset(val_data_tokens,token2id)
test_data_indices = token2index_dataset(test_data_tokens,token2id)

train_dataset = YelpDataset(train_data_indices, train_data_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=args.batch_size,
                                           collate_fn=yelp_collate_func,
                                           shuffle=True)

val_dataset = YelpDataset(val_data_indices, val_data_labels)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=args.batch_size,
                                           collate_fn=yelp_collate_func,
                                           shuffle=True)

test_dataset = YelpDataset(test_data_indices, test_data_labels)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=args.batch_size,
                                           collate_fn=yelp_collate_func,
                                           shuffle=False)




#print(len(id2token))
if args.model=='BOW':
    model = BagOfWords(len(id2token),args.hidden_size, args.emb_dim).to(device)

learning_rate = 0.001
num_epochs = 10 # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()  

if args.optimizer=='adam':
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
        data_batch, length_batch, label_batch = data.to(device), lengths.to(device), labels.to(device)
        #print(data_batch,label_batch,length_batch)
        outputs = F.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)

val_acc_list=[]
train_acc_list=[]
max_acc=0
for epoch in range(num_epochs):
    #linear annealing of learning rate at every 4th epoch
    if epoch%3==2:
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate*0.5)
    for i, (data, lengths, labels) in enumerate(train_loader):
        model.train()
        data_batch, length_batch, label_batch = data.to(device), lengths.to(device), labels.to(device)
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
            if val_acc>max_acc:
                torch.save(model.state_dict(), 'model/'+args.model)
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


