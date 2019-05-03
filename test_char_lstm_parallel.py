'''
Character-Level LSTM in PyTorch

In this code, I'll construct a character-level LSTM with PyTorch. The network will train
character by character on some text, then generate new text character by character.
This model will be able to generate new text based on the text from any provided book!

This network is based off of Udacity RNN mini project and which is in turn based off of Andrej Karpathy's
post on RNNs and implementation in Torch.

Below is the general architecture of the character-wise RNN.
'''

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib as mpl
mpl.use('Agg')    
import matplotlib.pyplot as plt

## Pre-processing the data
# one-hot encodeing
def one_hot_encode(arr, n_labels):
    '''
    Each character is converted into an integer (via our created dictionary) and
    then converted into a column vector where only it's corresponding integer
    index will have the value of 1 and the rest of the vector will be filled with 0's.

    Arguments:
        arr: An array of integers to be encoded, a Numpy array of shape ->
        (batch_size, sequence_length)
        n_labels: Dimension of the converted vector, an integer
    Returns:
        ont_hot: Encoded vectors that representing the input arr integers, a
                 Numpy array of shape ->
                 (batch_size, sequence_length, n_labels)
    '''
    # Initialize the encoded array
    # at this stage of shape (batch_size * sequence_length, n_labels)
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.0

    # Finally reshape it to get back the original array (but each element changed from
    # integer to an array, or an one-hot vector with shape)
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


## Create training batches
def get_batches(arr, batch_size, seq_len):
    '''
    Create a generator that returns batches of size batch_size x seq_len from arr.
    If:
        N --> batch size
        M --> number of time steps in a sequence (sequence length)
        K --> total number of batches
        L --> total length of text
    Then has a relationship:
        L = N * M * K

    Arguments:
        arr: Array of encoded text
        batch_size: Number of sequence per batch
        seq_len: Number of encoded chars in a sequence

    Returns:
        yield a batch
    '''
    # total number of full batches
    n_batches = len(arr) // (batch_size * seq_len)

    # keep only enough characters to make full batches
    arr = arr[:batch_size * seq_len * n_batches]

    # reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # change to dtype int64 (long), otherwise will occur RuntimeError in lstm training:
    # Expected object of scalar type Long but got scalar type Int for argument #2 'target'
    arr = arr.astype('int64')

    # iterate over the batches using a window of size seq_len
    for n in range(0, arr.shape[1], seq_len):
        x = arr[:, n: n+seq_len]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, n+seq_len]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


## LSTM Model
class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        '''
        Initialize CharRNN model.

        Arguments:
            tokens: Number of unique characters, or volume of vocabulary
            n_hidden: Number of neurons in a hidden layer
            n_layers: Number of hidden layers in RNN
            drop_prob: Dropout rate
            lr: Learning rate
        '''
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # input_size is the total number of characters (as the vector length)
        input_size = len(self.chars)
        
        # define the LSTM layer
        self.lstm = nn.LSTM(input_size, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        
        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # define a fully connected layer
        self.fc = nn.Linear(n_hidden, input_size)


    def forward(self, x, hidden):
        '''
        Forward pass through the network.
        These inputs are x, and the hidden/cell state.

        Arguments:
            x: Shaped (seq_len, batch, input_size)
            hidden: Shaped (num_layers * num_directions, batch, hidden_size)

        Returns:
            out: Shaped (seq_len, batch, num_directions * hidden_size)
            hidden: Shaped (num_layers * num_directions, batch, hidden_size)
        '''
        # Get LSTM outputs
        # reshape hidden state
        hidden = tuple([h.permute(1, 0, 2).contiguous() for h in hidden])

        lstm_out, hidden = self.lstm(x, hidden)

        # add dropout layer
        out = self.dropout(lstm_out)

        # shape the output to be (batch_size * seq_len, hidden_dim)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)

        # return the final output and the hidden state
        hidden = tuple([h.permute(1, 0, 2).contiguous() for h in hidden])
        return out, hidden


    def init_hidden(self, batch_size):
        '''
        Initialize hidden state.
        Create two new tensors with sizes n_layers x batch_size x n_hidden,
        initialized to zero, for hidden state and cell state of LSTM

        Arguments:
            batch_size: batch size, an integer

        Returns:
            hidden: hidden state initialized
        '''
        weight = next(self.parameters()).data
        if train_on_multi_gpus or train_on_gpu:
            hidden = (weight.new(batch_size, self.n_layers, self.n_hidden).zero_().cuda(),
                      weight.new(batch_size, self.n_layers, self.n_hidden).zero_().cuda())

        else:
            hidden = (weight.new(batch_size, self.n_layers, self.n_hidden).zero_(),
                      weight.new(batch_size, self.n_layers, self.n_hidden).zero_())

        return hidden


# Utility to plot learning curve
def loss_plot(losses, valid_losses):
    '''
    Plot the validation and training loss.
    
    Arguments:
        losses: A list of training losses
        valid_losses: A list of validation losses
    Returns:
        No returns, just plot the graph.
    '''
    # losses and valid_losses should have same length
    assert len(losses) == len(valid_losses)
    epochs = np.arange(len(losses))
    # plt.plot(epochs, losses, 'r-', valid_losses, 'b-')


# train the model!
def train(net, data, epochs=10, batch_size=16, seq_len=50, lr=0.001, clip=5, val_frac=0.1, every=10):
    '''
    Training a network and serialize it to local file.
    
    Arguments:    
        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_len: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        every: Number of steps for printing training and validation loss
    '''
    # training mode
    net.train()

    # define optimizer and loss
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # creating training and validation data
    val_idx = int(len(data) * (1 -val_frac))
    data, val_data = data[: val_idx], data[val_idx:]

    # count total time steps:
    counter = 0

    try:
        n_chars = len(net.chars)
    except AttributeError:
        # using DataParallel wrappper to use multiple GPUs
        n_chars = len(net.module.chars)

    if train_on_multi_gpus:
        net = torch.nn.DataParallel(net).cuda()
    elif train_on_gpu:
        net = net.cuda()
    else:
        print('Training on CPU...')

    # list to contain losses to be plotted
    losses = []
    vlosses = []

    for e in range(epochs):
        # initialize hidden state
        try:
            hidden = net.init_hidden(batch_size)
        except:
            # if using DataParallel wrapper to use multiple GPUs
            hidden = net.module.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_len):

            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu or train_on_multi_gpus:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise we'd backprop
            # through the entire training history.
            hidden = tuple([each.data for each in hidden])

            # zero acculated gradients
            net.zero_grad()

            # reshape inputs shape because not using batch_first=True as in solution code:
            ## inputs = inputs.permute(1, 0, 2)
            ## inputs = inputs.contiguous()

            # get output from model
            output, hidden = net(inputs, hidden)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_len))
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
            nn.utils.clip_grad_norm(net.parameters(), clip)
            opt.step()

            # loss stats
            if counter % every == 0:
                # get validation loss
                try:
                    val_h = net.init_hidden(batch_size)
                except AttributeError:
                    # if using DataParallel wrapper to use multipl GPUs
                    val_h = net.module.init_hidden(batch_size)
                val_losses = []
                net.eval()

                for x, y in get_batches(val_data, batch_size, seq_len):
                    # ont-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)

                    val_h = tuple([each.data for each in val_h])
                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                    if train_on_gpu or train_on_multi_gpus:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_len))
                    val_losses.append(val_loss.item())

                    # append loss into losses list
                    losses.append(loss)
                    vlosses.append(np.mean(val_losses))

                net.train()
                print(f'Epoch: {e+1}/{epochs}...',
                      f'Step: {counter}...',
                      f'Loss: {loss.item():.4f}...',
                      f'Val Loss: {np.mean(val_losses):.4f}')


    

    # decide file name by hyperparameters
    try:
        n_hidden = net.n_hidden
        n_layers = net.n_layers
        tokens = net.chars
        state_dict = net.state_dict()
    except AttributeError:
        # using DataParallel
        n_hidden = net.module.n_hidden
        n_layers = net.module.n_layers
        tokens = net.module.chars
        state_dict = net.module.state_dict()

    checkpoint = {'n_hidden': n_hidden,
                  'n_layers': n_layers,
                  'state_dict': state_dict,
                  'tokens': tokens}

    model_name = f'RNN-{n_hidden}-{n_layers}-{batch_size}-{seq_len}-{lr}-{clip}.net'

    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)

    # plot loss curve
    loss_plot(losses, vlosses)


# predict using the trained model and top-k sampling
def predict(net, char, hidden=None, top_k=None):
    '''
    Predict next character given the trained model and a starting sequence.

    Parameters:
        net: The training model
        char: A character
        hidden: Hidden state
        top_k: Choose a K (integer) to decide some K most probable characters

    Returns:
        The encoded value of the predicted char and the hidden state

    '''
    # tensor inputs
    
    try:
        num_tokens = len(net.chars)
        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, num_tokens)
    except AttributeError:
        # using DataParallel
        x = np.array([[net.module.char2int[char]]])
        x = one_hot_encode(x, num_tokens)

    # conver numpy array to torch tensor
    inputs = torch.from_numpy(x)

    if train_on_gpu or train_on_multi_gpus:
        inputs = inputs.cuda()

    # detach hidden state from history
    hidden = tuple([each.data for each in hidden])
    # get output of model
    output, hidden = net(inputs, hidden)
    # get output probabilities
    p = F.softmax(output, dim=1).data

    if train_on_gpu or train_on_multi_gpus:
        # move p back to cpu to use numpy
        p = p.cpu()

    # get top characters
    if top_k is None:
        top_ch = np.arange(num_tokens)
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    # return the encoded value of the predicted char and the hidden state
    try:
        string = net.int2char[char]
    except AttributeError:
        # using DataParallel
        string = net.module.int2char[char]

    return string, hidden


# Sample
def sample(net, size, prime='The', top_k=None):
    '''
    Sample a paragraph.

    Parameters:
        net: Trained model
        size: Length to be sampled
        prime: Starting words
        top_k: Use top_k or not

    Returns:
        A paragraph of sampled string
    '''
        
    if train_on_gpu or train_on_multi_gpus:
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]

    try:
        hidden = net.init_hidden(1)
    except AttributeError:
        # using DataParallel
        hidden = net.module.init_hidden(1)

    for ch in prime:
        char, hidden = predict(net, ch, hidden, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, hidden = predict(net, chars[-1], hidden, top_k=top_k)
        chars.append(char)

    return ''.join(chars)




if __name__ == '__main__':

    ## We'll load text file and convert it into integers for our network to use.
    # open text file and read data as 'text'
    with open('data/stone.txt', 'r') as f:
        text = f.read()


    ## Tokenization
    # encode the text and map each character to an interger and vice versa
    # create two dictionaries:
    # 1. int2char, maps integers to characters
    # 2. char2int, maps characters to integers
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # encode the text to integer array
    encoded = np.array([char2int[ch] for ch in text])

    # CUDA semantics:
    train_on_gpu = torch.cuda.is_available()
    train_on_multi_gpus = (torch.cuda.device_count() >= 2)
    gpus = torch.cuda.device_count()

    if train_on_multi_gpus:
        print(f"Tranning on {gpus} GPUs!")

    elif train_on_gpu:
        print('Training on GPU!')

    else: 
        print('No GPU available, training on CPU; consider making n_epochs very small.')


    # hyperparameters:
    n_hidden = 1024
    n_layers = 2
    batch_size = 32
    seq_len = 256
    n_epochs = 20
    lr = 0.001
    clip = 5


    # instantiate model
    net = CharRNN(chars, n_hidden, n_layers)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')


    # train the model
    # Comment the line below if already trained, will add interface in the future.
    train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_len=seq_len, lr=lr, clip=clip, every=100)


    # Load the trained. The saved model name is saved in the format of
    # 'RNN-{n_hidden}-{n_layers}-{batch_size}-{seq_len}-{lr}-{clip}.net'
    # Here we have loaded in a model that trained over 20 epochs `rnn_20_epoch.net`
    with open(f'RNN-{n_hidden}-{n_layers}-{batch_size}-{seq_len}-{lr}-{clip}.net', 'rb') as f:
        checkpoint = torch.load(f)
        
    loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    loaded.load_state_dict(checkpoint['state_dict'])

    # sample from the trained model to obtain a novel!
    text = sample(loaded, 1000, top_k=5, prime='the')
    print(text)
