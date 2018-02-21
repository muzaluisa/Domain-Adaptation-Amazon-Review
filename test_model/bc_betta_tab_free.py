# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:41:25 2017
Simple hierarchical sentence model
@author: Luiza
"""

import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import argparse
import os
from torch.nn.utils.rnn import pack_padded_sequence
import sys
from datetime import date
from preprocess_book_corpus import BookCorpus
from sklearn.neighbors import KDTree
from collections import Counter

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))


parser = argparse.ArgumentParser(description='Hierarchical alpha model')
parser.add_argument('-data-folder', type=str, default='/homeappl/home/sayfull1/NYC', help='directory of the book corpus [default: /homeappl/home/sayfull1/NYC]')
parser.add_argument('-batch-size', type=int, default=64, help = 'batch size [default: 1]')
parser.add_argument('-epochs', type=int, default=5000, help='number of epochs [default: 10]')
parser.add_argument('-dropout', type=float, default=0.2, help="dropout [default=0.5]")
parser.add_argument('-lr', type=float, default=0.001, help="learning rate [default=0.001]")
parser.add_argument('-hidden-size', type=int, default=200, help="the size of the sentence representations")
parser.add_argument('-K', type=int, default=1, help="the number of sentence representations [default=3]")
parser.add_argument('-optimizer', type=str, default='Adam', help="the name of the optimizator [default=Adam]")
parser.add_argument('-save-model', type=bool, default=False, help='wheather to save .pkl file of the model')
args = parser.parse_args()
print args

bc = BookCorpus(args.data_folder, small_dataset=False)
batch_size = args.batch_size
embed_matrix = bc.embed_matrix
Xtrain, Lenseq, Xtrain_ind = bc.get_data_with_word_indices(bc.Xtrain)

Xtrain_np =  np.array(Xtrain, dtype=np.int64)
Ntrain = len(Xtrain_np)
Xtrain = torch.from_numpy(Xtrain_np)
train_dataset = torch.utils.data.TensorDataset(Xtrain,Xtrain)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

num_epochs = args.epochs
learning_rate = args.lr
dropout = args.dropout

dim = 200
num_words = np.shape(Xtrain_np)[1]
input_size = dim
hidden_size = args.hidden_size
num_layers = args.K
voc_size = len(bc.vocabulary)
lr = args.lr


def get_bow_encoding(idx, sentence_lens):

    '''Given an array with word positions idx
    Returns its bow encoding
    '''

    num_batch = len(sentence_lens)
    res = np.zeros((num_batch, voc_size))
    for b in range(num_batch):
        ind = idx[b,0:sentence_lens[b]] # make sure its not the list
        res[b, ind] = 1
    return res

class HierarchicalSent(nn.Module):

    def init_hidden(self, batch_size_=args.batch_size):

        ''' Before we've done anything, we dont have any hidden state.
            Refer to the Pytorch documentation to see exactly why they have this dimensionality.
            The axes semantics are (num_layers, minibatch_size, hidden_dim)
        '''

        return (Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size).cuda()),
                Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size).cuda()))

    def __init__(self):

        super(HierarchicalSent, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_layers = num_layers
        self.K = args.K # number of hidden sentence representations, including bag of embeddings
        self.embed = nn.Embedding(voc_size, dim)
	
	self.decoder_form = "CONCATENATED_LINEAR" # "CONCATENATED_LINEAR", "RNN"
        self.enc = nn.ModuleList()
        self.enc.append(nn.Linear(dim, self.hidden_size))

        for i in range(self.K-1):
            self.enc.append(nn.Linear(dim + self.hidden_size, self.hidden_size))
        self.dropout = nn.Dropout(args.dropout)
        self.dec = nn.ModuleList()
        self.dec_weights = nn.Linear(voc_size, voc_size)
        self.dec.append(nn.Linear(dim, voc_size))
	self.output1 = nn.Linear((self.K+1)*dim, voc_size)
	self.out_lstm = nn.Linear(2*dim, voc_size)
	self.dec_lstm = nn.LSTM(dim, dim)
        for i in range(self.K-1):
                self.dec.append(nn.Linear(self.hidden_size, voc_size))
        self.embed.weight.data = torch.from_numpy(np.array(embed_matrix, dtype=np.float32))

    def forward(self, x, offsets):

        dtype = torch.cuda.FloatTensor
        h = []
        h.append(torch.mean(self.embed(x), dim=1))
        abs_offsets = map(abs, offsets)
	abs_offsets = map(lambda x: x+1, abs_offsets)
	num_batch = x.size()[0]
	abs_offsets_tensor = torch.from_numpy(np.array(abs_offsets, dtype=np.float32)).cuda().unsqueeze(1).expand(num_batch,voc_size)
	#print type(abs_offsets_tensor)
        # just encode on all levels and then take what is needed only
        for i in range(self.K):
            if i == 0:
                h_temp = self.enc[i](h[0])
            else:
                h_temp = self.enc[i](torch.cat((h[0],h[i]),1 ))
            h.append(torch.nn.functional.tanh(self.dropout(h_temp)))
        

	if self.decoder_form == "MEAN":
            dec_output = Variable(torch.zeros(self.K + 1, num_batch, voc_size).type(dtype))
            for b in range(num_batch):
                for layer in range(abs_offsets[b]):
		    dec_output[layer, b, :] = self.dec[layer](h[layer][b,:])
            dec_out = torch.sum(dec_output, 0) / Variable(abs_offsets_tensor)
			
	if self.decoder_form == "CONCATENATED_LINEAR":
	    encoder_outputs = Variable(torch.zeros(self.K + 1, num_batch, dim).type(dtype))
            for b in range(num_batch):
                for layer in range(abs_offsets[b]):
		    encoder_outputs[layer, b, :] = h[layer][b,:]
	    dec_out = self.output1(encoder_outputs.view(num_batch, dim * (self.K + 1)))

	if self.decoder_form == "RNN":
	    encoder_outputs = Variable(torch.zeros(self.K + 1, num_batch, dim).type(dtype))
            for b in range(num_batch):
                for layer in range(abs_offsets[b]+1):
                    encoder_outputs[layer, b, :] = h[layer][b,:]
	    _ , hn = self.dec_lstm(encoder_outputs.view(num_batch, self.K+1,dim), abs_offsets)
	    rnn_output = torch.cat((hn[0][0],hn[1][0]),0)
	    dec_out =  self.out_lstm(rnn_output)	
 
	word_prob = torch.sigmoid(dec_out)
        return word_prob

    def get_kth_sentence_encoding(self, word_idx_array, K):

        ''' Custom function for getting Kth
        sentence encoding
        Input:   word_idx_array - an array of word indices encoding a set of sentences
                 K - the level of embedding we would like to get for our sentences
        Returns:
                the list of embedding vectors
        '''
        
        h = []
        h.append(torch.mean(self.embed(word_idx_array), dim=1))
       
        for i in range(K):
            if i == 0:
                h_temp = self.enc[i](h[0])
            else:
                h_temp = self.enc[i](torch.cat((h[0],h[i]),1 ))
            h.append(torch.nn.functional.tanh(h_temp))
            
        #print 'Embedding size:', h[-1].size()
        return h[-1]
    
    def get_kth_sentence_encoding_numpy(self, word_idx_array_np, K):

        ''' Custom function for getting Kth
        sentence encoding
        Input:   word_idx_array - a numpy array of word indices encoding a set of sentences
                 K - the level of embedding we would like to get for our sentences
        Returns:
                the list of embedding vectors
        '''
        
        h = []
        word_idx_array = Variable(torch.from_numpy(word_idx_array_np)).cuda()
        
        h.append(torch.mean(self.embed(word_idx_array), dim=1))
        abs_offsets = map(abs, offsets)
        for i in range(K):
            if i == 0:
                h_temp = self.enc[i](h[0])
            else:
                h_temp = self.enc[i](torch.cat((h[0],h[i]),1 ))
            h.append(torch.nn.functional.tanh(h_temp))
            
        #print 'Embedding size:', h[-1].size()
        return h[-1].data.cpu().numpy()
    
    
best_valid_acc = -np.inf
valid_acc_history = []
curr_acc = -np.inf
epochs_new_lr = 0
train_loss_avg = 0

hs = HierarchicalSent()
hs.cuda()

if args.optimizer == "Adadelta":
   optimizer = torch.optim.Adadelta(hs.parameters(), lr=lr, rho=0.9)

if args.optimizer == "SGD":
    optimizer = torch.optim.SGD(hs.parameters(), lr=0.01, momentum=0.9)

if args.optimizer == "Adam":
    optimizer =  torch.optim.Adam(hs.parameters(), lr=lr)

model_saved = False
sampled_x = []
all_offsets = []
for epoch in range(args.epochs):
    optimizer.zero_grad()
    x_ind = np.zeros(batch_size, dtype=np.int64)
    offsets = np.zeros(batch_size, dtype=np.int64)
    for b in range(batch_size):
        x_pos = np.random.randint(0, Ntrain)
	if hs.K > 1:
        	offset = np.random.randint(1, hs.K) * np.random.choice([1, -1])
	else:
		offset = np.random.choice([1, -1])

        while not (Ntrain > (x_pos + offset) >=0):
            x_pos = np.random.randint(0, Ntrain)
	    if hs.K > 1:
            	offset = np.random.randint(1, hs.K) * np.random.choice([1, -1])
	    else:
		offset = np.random.choice([1, -1])

        x_ind[b] = x_pos
        offsets[b]= offset

    sampled_x.extend(x_ind)
    all_offsets.extend(offsets)
    sentence = Variable(Xtrain[x_ind, :]).cuda()
    neighbour = Variable(Xtrain[(x_ind + offsets), :]).cuda()

    outputs = hs(sentence, offsets)

    if np.random.randint(0,100) == 5:
        print 'sampled sentence:', bc.Xtrain[Xtrain_ind[x_pos]]
        print 'sampled neighbour sentence:', bc.Xtrain[Xtrain_ind[x_pos + offset]]
        print 'offset:', offset
	print 'output probs:', outputs.data.cpu().numpy()
        probs , indices = torch.sort(outputs, dim=1, descending=True)
        indices_np = indices.data.cpu().numpy()
        probs_np = probs.data.cpu().numpy()
        most_prob_words = []
        for index in indices_np[0][0:100]:
            most_prob_words.append(bc.vocabulary[index])

        print "with stop words: ", " ".join(most_prob_words)
   

    Xtrain_idx = np.array(Xtrain_np[x_ind + offsets])
    target = Variable(torch.cuda.FloatTensor(get_bow_encoding(Xtrain_idx, Lenseq[x_ind + offsets])))
    
    if np.random.randint(0,100) == 5:
	print 'target and output vector comparison'
	print target.data.cpu().numpy()[0]
	print outputs.data.cpu().numpy()[0]
    

    loss = -(target*torch.log(outputs+1e-6) + (1-target)*torch.log(1-outputs+1e-6)).mean(1).mean(0)
    loss.backward()
    optimizer.step()
    train_loss_avg+=loss.data[0]
    if epoch and epoch % 100 == 0:
        print epoch, train_loss_avg / 100.0
        train_loss_avg = 0
        
    if curr_acc > best_valid_acc:
	print 'The model is saved'
        best_valid_acc = curr_acc
        best_model = lstm.state_dict()
        best_opt = optimizer.state_dict()
        best_epoch = epoch
	model_saved = True
        d = date.today()
        torch.save({
                'epoch': epoch,
                'model': lstm.state_dict(),
                'optimizer': optimizer.state_dict()
                }, 'hierarchical_betta_' + d.isoformat() + '.pkl')


print Counter(sampled_x)[0:50]

hs1 = HierarchicalSent()
hs1.cuda()
optimizer = torch.optim.Adadelta(hs1.parameters(), lr=lr, rho=0.9)
d = date.today()
args.model_filename = 'hierarchical_betta_' + d.isoformat() + '.pkl'

if os.path.isfile(args.model_filename):
    print("=> loading checkpoint '{}'".format(args.model_filename))
    checkpoint = torch.load(args.model_filename)
    args.start_epoch = checkpoint['epoch']
    hs1.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.model_filename, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.model_filename))
    hs1 = hs   
# finding the nearest neigbours in the dataset

from sklearn.neighbors import KDTree

writer = csv.writer(open('KNN_results_hierarchical_sent.csv','a'), delimiter=',', lineterminator='\n')
writer.writerow(('index_sample', 'index_closest_neighbour','sampled sentence','nearest neighbour sentence'))

import scipy

for K in range(0, args.K+1):
    print 'Level:', K
    writer.writerow(( ("Level:" + str(K))))	
    hk = hs1.get_kth_sentence_encoding_numpy(Xtrain_np[0:5000], K)
    print 'MIN vector value:', np.min(hk)
    print 'MAX vector value:', np.max(hk)
    print 'MEAN vector value:', np.mean(hk)
    print 'Sample vector:', hk[0,:]     
    kdt = KDTree(hk, leaf_size=30, metric='euclidean')
    knn = kdt.query(hk, k=2, return_distance=False) 
    for element in knn[0:20]:
	writer.writerow((element[0], element[1], scipy.spatial.distance.cosine(hk[element[0]],hk[element[1]]), bc.Xtrain[element[0]],  bc.Xtrain[element[1]]))
        


