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
print(device)

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--data_dir', metavar='N', type=str,
                    help='Data directory',default='data')
parser.add_argument('--pretrained_vector_dir', metavar='N', type=str,
                    help='Pretrained vector directory',default='/scratch/ssp573/CloudML/wiki-news-300d-1M.vec')
parser.add_argument('--hidden_size_cnn', metavar='N', type=int,
                    help='Hidden layer size for CNN',default=512)
parser.add_argument('--hidden_size_linear',metavar='N',type=int,
                    help='Hidden layer size for linear layer', default=1024)
parser.add_argument('--kernel_size',metavar='N',type=int,
                    help='Kernel size for convolution', default=3)
parser.add_argument('--emb_dim', metavar='N', type=int,
                    help='Dimensions for word embedding',default=300)
parser.add_argument('--max_vocab_size', metavar='N', type=int,
                    help='Maximum vocabulary size',default=10000)
parser.add_argument('--batch_size', metavar='N', type=int,
                    help='Batch size',default=256)
parser.add_argument('--model',metavar='N', type=str,
                    help='Model',default='BOW')
parser.add_argument('--optimizer',metavar='N', type=str,
                    help='Optimizer',default='adam')
parser.add_argument('--pretrained_vecs',metavar='N', type=str,
                    help='Pretrained vectors (y/n)',default="y")
parser.add_argument('--model_name',metavar='N', type=str,
                    help='stemming (y/n)',default="model")
parser.add_argument('--lr',metavar='N', type=float,
                    help='learning_rate',default=0.001)
args = parser.parse_args()

train_tokenized=pd.read_pickle(args.data_dir+'/train_restaurants_tokenized_no_stem.pkl')#[:2000]
val_tokenized=pd.read_pickle(args.data_dir+'/val_restaurants_tokenized_no_stem.pkl')#[:2000]
test_tokenized= pd.read_pickle(args.data_dir+'/test_restaurants_tokenized_no_stem.pkl')#[:2000]

train_tokenized['sentiment']=np.where(train_tokenized['sentiment']=='pos',1,0)
val_tokenized['sentiment']=np.where(val_tokenized['sentiment']=='pos',1,0)
test_tokenized['sentiment']=np.where(test_tokenized['sentiment']=='pos',1,0)

train_tuples= [tuple(x) for x in train_tokenized[['text','sentiment']].values]
train_data_tokens,train_data_labels=zip(*train_tuples)

val_tuples= [tuple(x) for x in val_tokenized[['text','sentiment']].values]
val_data_tokens,val_data_labels=zip(*val_tuples)

test_tuples= [tuple(x) for x in test_tokenized[['text','sentiment']].values]
test_data_tokens,test_data_labels=zip(*test_tuples)

pretrained_vecs=None
if args.pretrained_vecs=='y':
    print("in1")
    token2id, id2token, pretrained_vecs = build_vocab_pretrained(args.pretrained_vector_dir)

train_data_indices = token2index_dataset(train_data_tokens,token2id)
val_data_indices = token2index_dataset(val_data_tokens,token2id)
test_data_indices = token2index_dataset(test_data_tokens,token2id)

train_dataset = YelpDataset(train_data_indices, train_data_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        collate_fn=yelp_collate_func_rnn,
                                        shuffle=True)

val_dataset = YelpDataset(val_data_indices, val_data_labels)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=args.batch_size,
                                        collate_fn=yelp_collate_func_rnn,
                                        shuffle=True)

test_dataset = YelpDataset(test_data_indices, test_data_labels)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=args.batch_size,
                                           collate_fn=yelp_collate_func_rnn,
                                           shuffle=False)

print("Model used: {}".format(args.model))

model= RNN(args.emb_dim,args.hidden_size_linear, args.hidden_size_cnn,len(id2token),args.pretrained_vecs, pretrained_vecs)

learning_rate = args.lr
print("learning_rate: {}".format(learning_rate))
num_epochs = 2 # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()

if args.optimizer=='adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
else:
    optimizer= torch.optim.SGD(model.parameters(), lr=learning_rate)

# Function for testing the model
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()

    for data, lengths, unsort_idx, labels in loader:
        data_batch, length_batch, unsort_batch, label_batch = data.to(device), lengths.to(device), unsort_idx.to(device), labels.to(device)
        #print(data_batch,label_batch,length_batch)
        outputs = F.softmax(model(data_batch, length_batch, unsort_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted).to(device)).sum().item()
    print("testing complete")
    return (100 * correct / total)

print("Starting training")
j = 0
time_point = []
time_string= []
val_acc_list=[]
train_acc_list=[]
max_acc=0

data_parallel = False
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model)
    data_parallel= True
    print("Using data parallel")
model.to(device)
# init_func = model.module.init_hidden(64)
for epoch in range(num_epochs):
    model.train()
    #linear annealing of learning rate at every 4th epoch
    if epoch%3==2:
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate*0.5)
    for i, (data, lengths, unsort_idx, labels) in enumerate(train_loader):
        data_batch, length_batch, unsort_batch, label_batch = data.to(device), lengths.to(device),unsort_idx.to(device), labels.to(device)
        optimizer.zero_grad()
        # import pdb; pdb.set_trace()
        # hidden.view(2, -1, args.hidden_size_cnn)
        outputs = model(data_batch, length_batch, unsort_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()
        # validate every 1000 iterations
        if i > 0 and i % 100 == 0:
            # validate
            val_acc = test_model(val_loader, model)
            train_acc= test_model(train_loader, model)
            if val_acc>max_acc:
                torch.save(model.state_dict(), 'model/'+args.model_name)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}, Training Acc: {}'.format(
                    epoch+1, num_epochs, i+1, len(train_loader), val_acc, train_acc))
            val_acc_list.append(val_acc)
            train_acc_list.append(train_acc)
import csv
import matplotlib as mpl
mpl.use('Agg')    
import matplotlib.pyplot as plt
#%matplotlib inline

#epochs=[i for i in range(1,num_epochs+1)]

plt.subplot(223)
plt.plot( time_point,train_acc_list, label="training accuracy")
plt.plot( time_point, val_acc_list, label="validation accuracy")
plt.ylabel("accuracy")
plt.xlabel("number of checks")
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig("./plots/{}_{}_{}_{}_{}_with_kernel_size:{}.png".format(args.model,args.hidden_size_cnn, learning_rate, args.batch_size, args.optimizer, args.kernel_size))
if model != 'RNN': # Just temp placeholder while I work out kinks
    file = open("./plots/raw_data_{}_{}_{}_{}_{}_with_kernel_size:{}.csv".format(args.model, args.hidden_size_cnn, learning_rate, args.batch_size, args.optimizer, args.kernel_size), 'w')
    with file:
        fnames = ['time_point', 'training_acc', 'validation_acc', 'time_string']
        writer = csv.DictWriter(file, fieldnames=fnames)  
        writer.writeheader()
        for i in range(len(time_point)):
            writer.writerow({'time_point' : str(time_point[i]), 'training_acc': str(train_acc_list[i]), 'validation_acc': str(val_acc_list[i]), 'time_string': time_string[i]})
