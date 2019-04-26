import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, hidden_size, emb_dim):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        # pay attention to padding_idx 
        #embed dimension should be atleast 100
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0).to(device)
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

class CNN(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, hidden_size,hidden_cnn, emb_dim):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(CNN, self).__init__()
        # pay attention to padding_idx 
        #embed dimension should be atleast 100
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0).to(device)
        self.conv1 = nn.Conv1d(emb_dim,hidden_cnn,kernel_size=3,padding=1)
        #self.conv2 = nn.Conv1d(hidden_cnn,hidden_cnn,kernel_size=3,padding=1)
        self.linear = nn.Linear(hidden_cnn,hidden_size)
        self.linear2=nn.Linear(hidden_size,2)
    
    def forward(self, data, length):
        #data.type()
        #print(data.type())
        batch_size, seq_len = data.size()
        embed = self.embed(data)
        hidden = self.conv1(embed.transpose(1,2)).transpose(1,2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        #hidden = self.conv2(hidden.transpose(1,2)).transpose(1,2)
        #hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        # return logits
        hidden=torch.max(hidden,dim=1)[0]
        hidden = F.relu(self.linear(hidden))
        out=self.linear2(hidden)
        return out
