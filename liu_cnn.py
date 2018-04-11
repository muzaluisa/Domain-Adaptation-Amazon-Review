# -*- coding: utf-8 -*-

"""
Created on Tue Nov 07 13:39:17 2017
fully shared model 
from Pengfei Liu paper "Adversarial Multi-task Learning for Text Classification"
@author: Luiza
"""

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from load_amazon_data import AmazonSentimentData
import torch.nn.functional as F
import argparse
import csv
import os
print 'CNN model'

parser = argparse.ArgumentParser(description='Vanilla LSTM model')
parser.add_argument('-dataset', type=str, default='baby', help='one of the amazon dataset names [default: baby]')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 50]')
parser.add_argument('-batch-size', type=int, default=100, help='batch size for training [default: 100]')
parser.add_argument('-dropout', type=float, default=0.2, help='the probability for dropout [default: 0.2]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of kernels per type')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
args = parser.parse_args()
print args

num_epochs = args.epochs
learning_rate = args.lr
embed_dim = 200
class_num = 2
dropout = args.dropout
num_words = 100
batch_size = args.batch_size
kernel_num = args.kernel_num

'''
STEP 1: LOADING DATASET
'''

DATA_FOLDER = '/wrk/sayfull1/NYC/mtl-dataset/mtl-dataset/'
amz = AmazonSentimentData(DATA_FOLDER, dataset_name=args.dataset, max_num_words=100)

(Xtrain, Ytrain, Lentrain), (Xtest, Ytest, Lentest) = amz.load_one_dataset_variable_length(args.dataset)

Nvalid = 200 #like in the pengfei liu paper

Xtrain, Xtest = np.array(Xtrain,dtype=np.float32), np.array(Xtest,dtype=np.float32)
Ytrain, Ytest = np.array(Ytrain,dtype=np.int64), np.array(Ytest,dtype=np.int64)


Xtrain = np.reshape(Xtrain,(-1,100,200))
Xtest = np.reshape(Xtest,(-1,100,200))

'''
STEP 2: MAKING DATASET ITERABLE
'''

train_data = torch.from_numpy(Xtrain[0:1000,:])
train_labels = torch.from_numpy(Ytrain[0:1000])

valid_data = torch.from_numpy(Xtrain[1000:,:])
valid_labels = torch.from_numpy(Ytrain[1000:])

test_data = torch.from_numpy(Xtest)
test_labels = torch.from_numpy(Ytest)

train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data,test_labels)
valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_labels)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class textCNN(nn.Module):
    
    def __init__(self):

        super(textCNN, self).__init__()        
        kernel_sizes = [2,3,4]
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, class_num)
        
    def forward(self, x):
        
        x = x.unsqueeze(1) # (N,Cin,H,W) to make Cin = 1
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] # [(N,Co,H), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]   # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1) # to concat across all kernels

        x = self.dropout(x) # (N,NUM_KERNEL_TYPES*NUM_CLASSES)
        logit = self.fc1(x) # (N,NUM_CLASSES)
        return logit        
        
cnn = textCNN()
cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

best_valid_acc = np.inf
valid_acc_history = []
# Train the Model
early_stopping = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
    
    
    correct = 0
    total = 0
    for images, labels in valid_loader:
    	images = Variable(images).cuda()
    	outputs = cnn(images)
    	_, predicted = torch.max(outputs.data, 1)
   	total += labels.size(0)
    	correct += (predicted.cpu() == labels).sum()
    curr_acc = correct*100.0/total
    valid_acc_history.append(curr_acc)
    print 'Current accuracy', curr_acc	     
    if curr_acc > best_valid_acc:
	best_valid_acc = curr_acc
        torch.save({
        	'epoch': epoch,
                'model': cnn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'accuracyHistory':valid_acc_history,
                'accuracy': best_valid_acc
            }, 'vanila_cnn_checkpoint')


# Test the Model
cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model: %d %%' % (100 * correct / total))
test_acc = 100 * correct / total

writer = csv.writer(open('cnn_results.csv','a'), delimiter=',', lineterminator='\n')
if os.stat("cnn_results.csv").st_size == 0:
	writer.writerow(( 'dataset','model name','number of feature maps', 'dropout', 'learning rate', 'num epochs', 'test acc'))
writer.writerow(( args.dataset, 'Vanilla CNN', args.kernel_num, args.dropout, args.lr, args.epochs,test_acc ))

# Save the Trained Model
#torch.save(cnn.state_dict(), 'cnn.pkl')

