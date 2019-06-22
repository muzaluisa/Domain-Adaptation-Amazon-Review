# -*- coding: utf-8 -*-

"""
Created on Tue Nov 07 13:39:17 2017
basic LSTM model 
with learned embeddings
from Pengfei Liu paper "Adversarial Multi-task Learning for Text Classification"
@author: Luiza
"""

import csv
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from load_amazon_data_preprocessed import AmazonSentimentData
import torch.nn.functional as F
import argparse
import os
from torch.nn.utils.rnn import pack_padded_sequence
import sys
from datetime import date

parser = argparse.ArgumentParser(description='Vanilla LSTM model')
parser.add_argument('-dataset', type=str, default='baby', help='one of the amazon dataset names [default: baby]')
parser.add_argument('-lr', type=float, default=1, help='initial learning rate [default: 1]')
parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train [default: 50]')
parser.add_argument('-batch-size', type=int, default=16, help='batch size for training [default: 100]')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.1]')
parser.add_argument('-hidden-size', type=int, default=50, help='number of units in the hidden space [default: 100]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
parser.add_argument('-num-layers', type=int, default=1, help ='Number of LSTM layers [default: 1]')
parser.add_argument('-ort-weights', type=bool, default=True, help ='If the weights of LSTM are orthogonal')
parser.add_argument('-early-stopping', type=bool, default=True, help ='If the model is taken from the best valid model')
parser.add_argument('-gradient-clipping-value', type=float, default=0, help ='Applies gradient clipping')
parser.add_argument('-max-seq-len', type=int, default=800, help ='Max sequece length [default: 800]')
parser.add_argument('-adaptive-learning-rate', type=bool, default=False, help='if learning rate decay is needed')
parser.add_argument('-model-date', type=str, default=date.today().isoformat(), help='date the model was run')
parser.add_argument('-var-len', type=bool, default=True, help='If the inputs should be variable length')
parser.add_argument('-save-model', type=bool, default=False, help='If to save the model')
parser.add_argument('-static', type=bool, default=True, help='If to train the word embeddings')
args = parser.parse_args()

args.adaptive_learning_rate = 0
print args

print 'lr:', args.lr
print 'hidden size:', args.hidden_size
print 'dataset:', args.dataset
print 'batch size', args.batch_size
print 'epochs:', args.epochs

print 'Vanilla LSTM model'
DATA_FOLDER = '/wrk/sayfull1/NYC/mtl-dataset/mtl-dataset/'
amz = AmazonSentimentData(DATA_FOLDER,dataset_name=args.dataset,max_num_words=args.max_seq_len)

(Xtrain, Ytrain, Lentrain),(Xvalid, Yvalid, Lenvalid), (Xtest, Ytest, Lentest) = \
    amz.load_one_dataset_variable_length(args.dataset)
voc_size = len(amz.vocabulary_set)

Nvalid = len(Xvalid) 
Ntrain = len(Xtrain) 
Ntest = len(Xtest)

print 'Ntrain:', Ntrain
print 'Nvalid:', Nvalid
print 'Ntest:', Ntest

print 'percent of positive classes in training dataset:'
print np.mean(Ytrain)

print 'percent of positive classes in test dataset:'
print np.mean(Ytest)

Xtrain, Xvalid, Xtest = np.array(Xtrain,dtype=np.float32), np.array(Xvalid,dtype=np.float32),\
 np.array(Xtest,dtype=np.float32)
Ytrain, Yvalid, Ytest = np.array(Ytrain,dtype=np.int64), np.array(Yvalid,dtype=np.int64),\
 np.array(Ytest,dtype=np.int64)

N, num_words, dim  = np.shape(Xtrain)

train_data = torch.from_numpy(Xtrain)
train_labels = torch.from_numpy(Ytrain)

test_data = torch.from_numpy(Xtest)
test_labels = torch.from_numpy(Ytest)

valid_data = torch.from_numpy(Xvalid)
valid_labels = torch.from_numpy(Yvalid)

train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data,test_labels)
valid_dataset = torch.utils.data.TensorDataset(valid_data,valid_labels)

batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

num_epochs = args.epochs
learning_rate = args.lr
dropout = args.dropout

input_size = dim
sequence_length = num_words
hidden_size = args.hidden_size
num_layers = 1
num_classes = 2
    
class LSTM(nn.Module):
   
    def init_hidden(self, batch_size_=args.batch_size):
        
        ''' Before we've done anything, we dont have any hidden state.
            Refer to the Pytorch documentation to see exactly why they have this dimensionality.
            The axes semantics are (num_layers, minibatch_size, hidden_dim)
        '''
        
        return (Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size).cuda()),
                Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size).cuda()))
    
    def get_ort_weight(self, m=hidden_size, n=input_size):
        
        return torch.nn.init.orthogonal(torch.FloatTensor(m,n))

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(voc_size, dim)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.hidden = self.init_hidden()

        if args.ort_weights:
            self.lstm.weight_ih_l0.data = torch.cat((self.get_ort_weight(),self.get_ort_weight(),self.get_ort_weight(),self.get_ort_weight()),0)    
            self.lstm.weight_hh_l0.data = torch.cat((self.get_ort_weight(hidden_size,hidden_size),\
            self.get_ort_weight(hidden_size,hidden_size),self.get_ort_weight(hidden_size,hidden_size),self.get_ort_weight(hidden_size,hidden_size)),0)
        
        self.embed.data = torch.from_numpy(np.array(amz.embed_matrix, dtype=np.float32))
        
    def forward(self, x):
        
        x = self.embed(x)
        #if args.static:
        #    x = Variable(x)
        out, self.hidden = self.lstm(x, self.hidden)
        res = self.dropout(torch.cat([self.hidden[0][0], self.hidden[1][0]], 1))
        out = self.fc(res)
        
        return out

lstm = LSTM(input_size, hidden_size, num_layers, num_classes)
lstm.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(lstm.parameters(), lr=args.lr, rho=0.9)

best_valid_acc = -np.inf
valid_acc_history = []
early_stopping = args.early_stopping
lr = args.lr
epochs_new_lr = 0

for epoch in range(num_epochs):
    
    print 'Epoch:', epoch
    train_loss_avg = 0

    idx =  np.array(np.random.permutation(range(Ntrain)))
    idx_torch = torch.LongTensor(idx)
    train_data = torch.index_select(train_data, 0, idx_torch)
    train_labels = torch.index_select(train_labels, 0, idx_torch)
    Lentrain = Lentrain[idx]

    for i in range(int(np.ceil(Ntrain//batch_size))):

        if (batch_size*(i+1)) <= Ntrain:
            images = train_data[batch_size*i:batch_size*(i+1)]
            labels = train_labels[batch_size*i:batch_size*(i+1)]
            lens = Lentrain[batch_size*i:batch_size*(i+1)]
        else:
            images = train_data[batch_size*i:]
            labels = train_labels[batch_size*i:]
            lens = Lentrain[batch_size*i:]
    
        ind = torch.LongTensor(np.argsort(np.array(lens))[::-1].copy())
        
        images = Variable(torch.index_select(images, 0, ind)).cuda()
        labels = Variable(torch.index_select(labels, 0, ind)).cuda()
        lens = sorted(lens)[::-1]  
        
        if args.var_len:
            x = pack_padded_sequence(images, lens, batch_first=True)
        else:
            x = images
    
        optimizer.zero_grad()
    
        if batch_size*(i+1) > Ntrain:
            lstm.hidden = lstm.init_hidden(Ntrain-batch_size*i)
        else:
            lstm.hidden = lstm.init_hidden()
    
        outputs = lstm(x)
        loss = criterion(outputs, labels)
        loss.backward()
    
        if args.gradient_clipping_value > 0:
            torch.nn.utils.clip_grad_norm(lstm.parameters(), args.gradient_clipping_value)
    
        optimizer.step()
        train_loss_avg+=loss.data[0]
     
    print 'Mean Cross Entropy loss:', train_loss_avg/len(train_loader)
    total = 0
    correct = 0
    lstm.eval()
    
    for i, (images, labels) in enumerate(valid_loader):
        images = Variable(images).cuda()

        if batch_size*(i+1) > Nvalid:
            lstm.hidden = lstm.init_hidden(Nvalid-batch_size*i)
        else:
            lstm.hidden = lstm.init_hidden()
    
        if args.var_len:
            if batch_size*(i+1) <= Nvalid:
                x = pack_padded_sequence(images, Lenvalid[i*batch_size:(i+1)*batch_size], batch_first=True)
            else:
                x = pack_padded_sequence(images, Lenvalid[i*batch_size:], batch_first=True)
    
            outputs = lstm(x)
        else:
            outputs = lstm(images)
    
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    
    curr_acc = correct*100.0/total
    valid_acc_history.append(curr_acc)
    print 'Current accuracy : ', curr_acc

    if curr_acc > best_valid_acc:
        best_valid_acc = curr_acc
        best_model = lstm.state_dict()
        best_opt = optimizer.state_dict()
        best_epoch = epoch
        torch.save({
                    'epoch': epoch,
                    'model': lstm.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'accuracyHistory': valid_acc_history,
                    'accuracy': best_valid_acc
                }, 'vanila_lstm_checkpoint')

    lstm.train()

print 'Best valid accuracy : ', best_valid_acc, ' from epoch: ', best_epoch
if args.early_stopping:
    print 'Loading the best model...'
    lstm.load_state_dict(best_model)
    optimizer.load_state_dict(best_opt)    

lstm.eval()
correct = 0
total = 0

for i, (images, labels) in enumerate(test_loader):
    images = Variable(images.cuda())
    if batch_size*(i+1) > Ntest:
        lstm.hidden = lstm.init_hidden(Ntest-batch_size*i)
    else:
        lstm.hidden = lstm.init_hidden()

    if args.var_len:
        if (batch_size*(i+1)) <= Ntest:
            x = pack_padded_sequence(images, Lentest[batch_size*i:batch_size*(i+1)], batch_first=True)
        else:
            x = pack_padded_sequence(images, Lentest[batch_size*i:], batch_first=True)

        outputs = lstm(x)
    else:
        outputs = lstm(images)

    _, predicted = torch.max(outputs.data, 1)
    total+= labels.size(0)
    correct+= (predicted.cpu() == labels).sum()

print('Test accuracy of the model: %.2f %%' % (100.0 * correct / total)) 
test_acc = (100 * correct) / total

writer = csv.writer(open('lstm_results_new.csv','a'), delimiter=',', lineterminator='\n')
if os.stat("lstm_results_new.csv").st_size == 0:
    writer.writerow(( 'date','dataset', 'model name', 'hidden units', 'dropout', 'num layers', 'learning rate',\
    'num epochs', 'gradient clipping value', 'early stopping', 'ort weights', 'max sequence len', \
    'adaptive learning rate', 'batch_size', 'test acc'))

d = date.today()
writer.writerow((d.isoformat(), args.dataset, 'Vanilla LSTM', args.hidden_size, args.dropout, \
args.num_layers, args.lr, args.epochs, args.gradient_clipping_value, args.early_stopping,\
args.ort_weights, num_words, args.adaptive_learning_rate, args.batch_size, test_acc ))


if args.save_model:
    torch.save(lstm.state_dict(), 'rnn.pkl')
