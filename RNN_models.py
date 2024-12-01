import argparse
import gc
import json
import glob
import os
import shutil
import sys
import numpy as np
import time
import collections
import math
from math import sqrt
from numpy import concatenate
import enum
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class Classification_RNN(nn.Module):
    def __init__(self, num_layers,hidden_dim, embedding_dim, batch_size, vocab_size_note, vocab_size_gpt4, dropout=0.2):
        super(Classification_RNN, self).__init__()
        self.n_layers = num_layers
        self.dim =  hidden_dim
        self.b_size =  batch_size
        self.v_note =  vocab_size_note
        self.v_gpt4 =  vocab_size_gpt4

        self.output_dim = 1 # Binary Classification
        self.hidden_dim = hidden_dim
 
        self.n_layers = num_layers
        self.vocab_size_n = vocab_size_note
        self.vocab_size_g = vocab_size_gpt4
    
        # embedding and LSTM layers
        self.embedding_note = nn.Embedding(self.vocab_size_n , embedding_dim)
        self.embedding_gpt4 = nn.Embedding(self.vocab_size_g , embedding_dim)
        
        #lstm
        self.lstm_note = nn.LSTM(input_size=embedding_dim, hidden_size=self.dim,
                                  num_layers=self.n_layers, batch_first=True)
                                  
        self.lstm_gpt4 = nn.LSTM(input_size=embedding_dim, hidden_size=self.dim,
                                  num_layers=self.n_layers, batch_first=True)
                                  
        # self.fc_note = nn.Linear(hidden_dim, self.output_dim)
        # self.fc_gpt_4 = nn.Linear(hidden_dim, self.output_dim)
        
        # dropout layer
        # self.dropout = nn.Dropout(0.2)
    
        # linear and sigmoid layer
        self.fc = nn.Linear((self.hidden_dim) * 2, self.output_dim)
        self.sig = nn.Sigmoid()
        
        
        

    def init_hidden(self, batch_size,device):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        # h0 = torch.rand((self.n_layers,self.b_size,self.dim)).to(device)
        # c0 = torch.rand((self.n_layers,self.b_size,self.dim)).to(device)


        h0 = torch.rand((self.n_layers,batch_size,self.dim)).to(device)
        c0 = torch.rand((self.n_layers,batch_size,self.dim)).to(device)

        # print('h0 size ** ',h0.size())
        # print('c0 size **',c0.size())
        hidden = (h0,c0)
        return hidden
        
    def forward(self, x,hidden):


        # print('input size >> ',x.size())
        batch_size = x.size(0)
        x_note = x[:,:332]
        x_gpt4 = x[:,332:]

        # print('x_note size >> ',x_note.size())
        # print('x_gpt4 size >> ',x_gpt4.size())
        # print(tess)
        

        # print('vocab_size_n size >> ',type(self.vocab_size_n))
        # print('vocab_size_g size >> ',type(self.vocab_size_g))
        
        # embeddings and lstm_out
        
        
        # embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        

        embeds_note = self.embedding_note(x_note)
        embeds_gpt_4 = self.embedding_gpt4(x_gpt4)


        # print('embeds_note size >> ',embeds_note.size())
        # print('embeds_gpt_4 size >> ',embeds_gpt_4.size())
        # print(tess)
            
        lstm_out_note, (hidden_note,cell_note) = self.lstm_note(embeds_note, hidden)
        
        lstm_out_gpt4, (hidden_gpt_4,cell_gpt4) = self.lstm_gpt4(embeds_gpt_4, hidden)

        # print('lstm_out_note size >> ',lstm_out_note.size())
        # print('lstm_out_gpt4 size >> ',lstm_out_gpt4.size())
        

        # print('hidden_note size >> ',hidden_note.size())
        # print('cell_note size >> ',cell_note.size())
        

        # print('hidden_gpt_4 size >> ',hidden_gpt_4.size())
        # print('cell_gpt4 size >> ',cell_gpt4.size())

        # print(tess)
        lstm_out_note = lstm_out_note.contiguous().view(-1, self.hidden_dim)        # Kaggle Sentiment Analysis using LSTM PyTorch
        lstm_out_gpt4 = lstm_out_gpt4.contiguous().view(-1, self.hidden_dim)        # Kaggle Sentiment Analysis using LSTM PyTorch

        # print('lstm_out_note size ** ',lstm_out_note.size())
        # print('lstm_out_gpt4 size ** ',lstm_out_gpt4.size())
       
       
        # dropout and fully connected layer
        # out = self.dropout(lstm_out)
        # out = self.fc(out)
        

        # print('hidden_gpt_4[0] :: ', hidden_gpt_4[0].size())
        # print('hidden_gpt_4[1] :: ', hidden_gpt_4[1].size())
        
        
        # print('hidden_note[-1] :: ', hidden_note[-1].size())
        # print('hidden_gpt_4[-1] :: ', hidden_gpt_4[-1].size())
        
        # out_note = self.fc_note(hidden_note[-1])
        # out_gpt_4 = self.fc_gpt_4(hidden_gpt_4[-1])
        
        
        # print('out_note size ** ',out_note.size())
        # print('out_gpt_4 size ** ',out_gpt_4.size())
        
        cat_hidden_gpt_4_notes = torch.concat((hidden_note, hidden_gpt_4),2)
        # print('cat_hidden_gpt_4_notes  ** ',cat_hidden_gpt_4_notes.size())

        # print('cat_hidden_gpt_4_notes  ** ',cat_hidden_gpt_4_notes[-1].size())

        out = self.fc(cat_hidden_gpt_4_notes[-1])

        # print('out  ** ',out.size())
        # out_gpt_4 = self.fc_gpt_4(hidden_gpt_4[-1])
        
        # sigmoid function
        sig_out = self.sig(out)
        # print(tess)
        

        # print('sig_out  * ',sig_out.size())
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        # print('sig_out  ** ',sig_out.size())
        # print(tess)

        # sig_out = sig_out[:, -1] # get last batch of labels
        # print('sig_out  *** ',sig_out.size())
        # print(tess)
        
        # return last sigmoid output and hidden state
        # return sig_out, hidden

        return sig_out        
        
