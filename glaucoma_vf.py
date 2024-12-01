import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import RNN_models
import os
import time
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from torchtext.datasets import IMDB
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# df_main = pd.read_csv('./clinical_notes.csv')   # ./ means innn the same directory as that of this python file

df_main = pd.read_csv('./clinical_notes.csv',index_col=False)   # ./ means innn the same directory as that of this python file




df_cat_keys={}
df_cat_keys['gender'] = df_main['gender'].unique()
df_cat_keys['race'] = df_main['race'].unique()
df_cat_keys['ethnicity'] = df_main['ethnicity'].unique()
df_cat_keys['language'] = df_main['language'].unique()
df_cat_keys['maritalstatus'] = df_main['maritalstatus'].unique()
df_cat_keys['glaucoma'] = df_main['glaucoma'].unique()

# we apply label encoding for the target variable column in this case it is Glaucoma

label_encoder = preprocessing.LabelEncoder()

df_main['target'] = label_encoder.fit_transform(df_main['glaucoma'])
print('after target >> ',df_main.shape)    #10000,12

print(df_main.columns)
# df_main.to_csv('./glaucoma.csv')

df_main = pd.read_csv('./glaucoma.csv',index_col=False)     # ./ means innn the same directory as that of this python file
print(df_main.shape)    # one extra column named target so we need to drop the column named glaucoma... where 1 ==> yes; 0==> No 
df_main = df_main.drop(columns = ['glaucoma'])
print('after dropping glaucoma >> ',df_main.shape)    #10000,12

# label encoding for the target variable ends here

print(df_main.shape)
print(df_main.columns)


# we remove 'glaucoma' from the dictionary too
del df_cat_keys['glaucoma']


####################     we apply one-hot encoding for a single variable    ####################

# encoder  = OneHotEncoder(sparse_output = False)
# Get one hot encoding of columns B


# one_hot_df = pd.get_dummies(df_main['race'])

# 3 ways to do this

# Method a
# Drop column 'race' as it is now encoded
# df_main = df_main.drop('race',axis = 1)
# Join the encoded df
# df_main_encoded = pd.concat([df_main,one_hot_df], axis=1)
# df_main_encoded.to_csv('./glaucoma_encoded_v1.csv')

# Method b
# Drop column 'race' as it is now encoded
# df_main = df_main.drop('race',axis = 1)
# Join the encoded df
# df_main_encoded = df_main.join(one_hot_df)
# df_main_encoded.to_csv('./glaucoma_encoded_v2.csv')


# Method c
# combine all in one instruction

# df_main_encoded =  df_main.drop('race',axis = 1).join(one_hot_df)
# df_main_encoded.to_csv('./glaucoma_encoded_v3_b.csv')


####################     we apply one-hot encoding for a several variables    ####################


one_hot_df = pd.get_dummies(df_main, columns=['race','gender','ethnicity','maritalstatus','language'])


# Does not work
# df_main_encoded =  df_main.drop(['race','gender','ethnicity','maritalstatus','language'],axis = 1)
# Join the encoded df
# df_main_encoded = df_main.join(one_hot_df)
# join(one_hot_df)
# df_main_encoded.to_csv('./glaucoma_encoded_v5.csv')
##########################################################

df_main_encoded =  df_main.drop(['race','gender','ethnicity','maritalstatus','language'],axis = 1)


df_main_encoded = pd.concat([df_main_encoded,one_hot_df], axis=1)


# df_main_encoded.to_csv('./glaucoma_encoded_v5_c.csv')
one_hot_df.to_csv('./glaucoma_encoded_v6_b.csv')



####################    one-hot encoding ends here  ####################

# post one-hot encoding we have the data in True False formats ....  we need this in a 0 1 format ... for this we use a Label Encoder

# iterate over the keys in the dictionary

for keys in df_cat_keys:
    print('>>>>>>>>>>>> ',keys)
    values = df_cat_keys[keys]
    for val in values:
        print('::::::::::::: ',val)
        col_name = keys+'_'+val
        col = one_hot_df[col_name]
        label_encoder = preprocessing.LabelEncoder()

        one_hot_df[col_name] = label_encoder.fit_transform(one_hot_df[col_name])
        


one_hot_df = one_hot_df.drop('Unnamed: 0', axis = 1)
one_hot_df.to_csv('./glaucoma_encoded_v7_b.csv')
print('\n','=========','\n')



# we now separate the data into 'training' 'validation' and 'test'  on the basis of 'use' column

df_one_hot_df_train = one_hot_df[one_hot_df['use'] == 'training']
df_one_hot_df_val = one_hot_df[one_hot_df['use'] == 'validation']
df_one_hot_df_test = one_hot_df[one_hot_df['use'] == 'test']



df_one_hot_df_train = df_one_hot_df_train.drop(columns = ['use'])
df_one_hot_df_val = df_one_hot_df_val.drop(columns = ['use'])
df_one_hot_df_test = df_one_hot_df_test.drop(columns = ['use'])


# # data separation ends here



note_train = df_one_hot_df_train['note']
note_val = df_one_hot_df_val['note']
note_test = df_one_hot_df_test['note']


cg_train = df_one_hot_df_train['gpt4_summary']
cg_val = df_one_hot_df_val['gpt4_summary']
cg_test = df_one_hot_df_test['gpt4_summary']




print('Longest Length chat_gpt4 train summary :: ', df_one_hot_df_train['gpt4_summary'].str.len().max())
print('Longest Length chat_gpt4 val summary :: ', df_one_hot_df_val['gpt4_summary'].str.len().max())
print('Longest Length chat_gpt4 test summary:: ', df_one_hot_df_test['gpt4_summary'].str.len().max())

print('Longest Length note train summary :: ', df_one_hot_df_train['note'].str.len().max())
print('Longest Length note val summary :: ', df_one_hot_df_val['note'].str.len().max())
print('Longest Length note test summary:: ', df_one_hot_df_test['note'].str.len().max())

# note_unclean_tr = df_one_hot_df_train['note']


# clean_text=text.lower().replace("\n", " ")
# clean_text=clean_text.replace("-", " ")

print(one_hot_df['note'].iloc[0])
print('\n','=========','\n')
one_hot_df['note'] = one_hot_df['note'].apply(lambda x : x.lower().replace('\n',','))
print(one_hot_df['note'].iloc[0])
print('\n','=========','\n')
one_hot_df['note'] = one_hot_df['note'].apply(lambda x : x.lower().replace('-',','))
print(one_hot_df['note'].iloc[0])
print('\n','=========','\n')


one_hot_df['note'] = one_hot_df['note'].apply(lambda x : x.split(' '))
print(one_hot_df['note'].iloc[0])
print('\n','=========','\n')




print(one_hot_df['gpt4_summary'].iloc[0])
print('\n','=========','\n')
one_hot_df['gpt4_summary'] = one_hot_df['gpt4_summary'].apply(lambda x : x.lower().replace('\n',','))
print(one_hot_df['gpt4_summary'].iloc[0])
print('\n','=========','\n')
one_hot_df['gpt4_summary'] = one_hot_df['gpt4_summary'].apply(lambda x : x.lower().replace('-',','))
print(one_hot_df['gpt4_summary'].iloc[0])
print('\n','=========','\n')


one_hot_df['gpt4_summary'] = one_hot_df['gpt4_summary'].apply(lambda x : x.split(' '))
print(one_hot_df['gpt4_summary'].iloc[0])
print('\n','=========','\n')

 
 
 






def replace_fn(clean_text):
    # print('type of data :: ', type(clean_text))
    # print('\n','<<<<<<<<<<<<<<<<<    clean text  >>>>>>>>>>>>>>>>','\n')


    # for x in ",.:;?!$()/_&%*@'`":
    #     clean_text=clean_text.replace(f"{x}", f" {x} ")
    # clean_text=clean_text.replace('"', ' " ')
    # text=clean_text.split()
    
    # print(clean_text)
    # new_text = []
    # print('\n','<<<<<<<<<<<<<<<<<    clean text  >>>>>>>>>>>>>>>>','\n')
    for idx,word in enumerate(clean_text):
        for x in ",.:;?!$()/_&%*@'`":
            word=word.replace(x, ' ' + x + ' ')
            # new_text.append(word)  
            clean_text[idx] = word
            # word=word.replace('/', ' / ')
            word=word.replace('"', ' " ')
            clean_text[idx] = word
          
    return clean_text         
        
        
one_hot_df['note'] = one_hot_df['note'].apply(replace_fn)

print(one_hot_df['note'].iloc[0])
print('::::: ',len(one_hot_df['note'].iloc[0]),' :::::')
print('\n','=========','\n')


one_hot_df['gpt4_summary'] = one_hot_df['gpt4_summary'].apply(replace_fn)

print(one_hot_df['gpt4_summary'].iloc[0])
print('::::: ',len(one_hot_df['gpt4_summary'].iloc[0]),' :::::')
print('\n','=========','\n')


print('Longest Length note dataset summary :: ', one_hot_df['note'].str.len().max())
print('\n','=========','\n')
print('Longest Length gpt4_summary dataset summary :: ', one_hot_df['gpt4_summary'].str.len().max())


# all_words_notes = ''.join(one_hot_df['note'])
# print(len(all_words_notes))




#############   note encoding and padding begins here   ################



text =[]
for notes in one_hot_df['note']:
    for word in notes:
        text.append(word)


print(len(text))


word_counts = Counter(text)
words = sorted(word_counts, key=word_counts.get, reverse= True)
print(words[:10])

text_length=len(text)
num_unique_words_note=len(words)
print(f"the text contains {text_length} words")
print(f"there are {num_unique_words_note} unique tokens")  

word_to_int={v:k for k,v in enumerate(words)} 
int_to_word={k:v for k,v in enumerate(words)}
print({k:v for k,v in word_to_int.items() if k in words[:10]})
print({k:v for k,v in int_to_word.items() if v in words[:10]})

print(text[0:20])
wordidx=[word_to_int[w] for w in text]  
print([word_to_int[w] for w in text[0:20]])  

print(type(int_to_word))

print(len(int_to_word))

print((int_to_word[0]))     #' , '
print((int_to_word[1]))     #'and'  


print((word_to_int[' , '])) # 0
print((word_to_int['and'])) # 1

# Encode all reviews as per the word2idx dictionary
encoded_review = [[word_to_int[word] for word in note] for note in one_hot_df['note']]
print(len(one_hot_df['note']))
print(len(encoded_review))


print('\n','>>>>>>>>','\n')

# pad_sequences and prepare dataloader from here 
padded_reviews_1 = pad_sequence([torch.tensor(review) for review in encoded_review])

padded_reviews_1 = padded_reviews_1.permute(1,0)
# print(padded_reviews_1.permute(1,0))


print(padded_reviews_1.size())    #torch.Tensor



def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features
    
seq_len = 332    
padded_reviews_2 = padding_(encoded_review,seq_len)    
print('padded_reviews_2 shape >',padded_reviews_2.shape) # numpy array
# padded_reviews_2 = np.expand_dims(padded_reviews_2, axis=1)
    

# print(type(padded_reviews)) 
print('padded_reviews_2 shape >>',padded_reviews_2.shape) # numpy array
# print(padded_reviews_2[:5])
# print(padded_reviews_2[0,100:200])


maxlen_note = padded_reviews_2.shape[1]
print('maxlen_note >> ', maxlen_note)

# lst_padded_reviews_2 = padded_reviews_2.tolist()
lst_padded_reviews_2 = padded_reviews_2.tolist()
print('lst_padded_reviews_2 >> ',type(lst_padded_reviews_2))
# print(len(encoded_review))
print(len(lst_padded_reviews_2))
# print(tess)




series1 = pd.Series(encoded_review, name='encoded_note') # not needed

series2 = pd.DataFrame(padded_reviews_2)

new_series = pd.concat([one_hot_df[['use','target']],series2],axis=1)

print(new_series.shape)

# print(tess)
# one_hot_df = pd.concat([one_hot_df,series1,series2],axis=1)

# print(one_hot_df.columns)
# print(one_hot_df.shape)


print('\n','<<<<<<<<<<<<<<<<<    >>>>>>>>>>>>>>>>>>','\n')


#############   note encoding and padding ends here   ################



#############   gpt4_summary encoding and padding begins here   ################




text =[]
for notes in one_hot_df['gpt4_summary']:
    for word in notes:
        text.append(word)



print(len(text))


word_counts = Counter(text)
words = sorted(word_counts, key=word_counts.get, reverse= True)
print(words[:10])

text_length=len(text)
num_unique_words_gpt4=len(words)
print(f"the text contains {text_length} words")
print(f"there are {num_unique_words_gpt4} unique tokens")  

word_to_int={v:k for k,v in enumerate(words)} 
int_to_word={k:v for k,v in enumerate(words)}
print({k:v for k,v in word_to_int.items() if k in words[:10]})
print({k:v for k,v in int_to_word.items() if v in words[:10]})

print(text[0:20])
wordidx=[word_to_int[w] for w in text]  
print([word_to_int[w] for w in text[0:20]])  

print(type(int_to_word))

print(len(int_to_word))





print((int_to_word[0]))     #' , '
print((int_to_word[1]))     #'and'  


print((word_to_int[' , '])) # 0
print((word_to_int['and'])) # 1

# Encode all reviews as per the word2idx dictionary
encoded_review = [[word_to_int[word] for word in note] for note in one_hot_df['gpt4_summary']]
print(len(one_hot_df['gpt4_summary']))
print(len(encoded_review))


print('\n','>>>>>>>>','\n')

# pad_sequences and prepare dataloader from here 
padded_reviews_1 = pad_sequence([torch.tensor(review) for review in encoded_review])

padded_reviews_1 = padded_reviews_1.permute(1,0)
# print(padded_reviews_1.permute(1,0))


print(padded_reviews_1.size())    #torch.Tensor




seq_len = 137    
padded_reviews_2 = padding_(encoded_review,seq_len)    


maxlen_gpt4 = padded_reviews_1.shape[1]
print('maxlen_gpt4 >> ', maxlen_gpt4)
    
# lst_padded_reviews_2 = padded_reviews_2.tolist()
print('<<<<<<< padded_reviews_2 >>>>>>>>> ',type(padded_reviews_2))
lst_padded_reviews_2 = padded_reviews_2.tolist()

print('<<<<<<< lst_padded_reviews_2 >>>>>>>>> ',type(lst_padded_reviews_2))

print('\n','>>>>>>>>','\n')



series2 = pd.DataFrame(padded_reviews_2)


new_series = pd.concat([new_series,series2],axis=1)

# print(new_series.columns)
print(new_series.shape)

# print(tess)

print('\n','<<<<<<<<<<<<<<<<<    >>>>>>>>>>>>>>>>>>','\n')


#############   gpt4_summary encoding and padding ends here   ################





X_train = new_series[new_series['use'] == 'training']
X_val = new_series[new_series['use'] == 'validation']
X_test = new_series[new_series['use'] == 'test']


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

X_train = X_train.drop(columns = ['use','target'])
X_val = X_val.drop(columns = ['use','target'])
X_test = X_test.drop(columns = ['use','target'])


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)



# X_train = X_train[['padded_encoded_note']]
# X_val = X_val[[ 'padded_encoded_note']]
# X_test = X_test[[ 'padded_encoded_note']]
#
print('\n','****************************','\n')


target_df = new_series[['target','use']]
# print(target_df.columns)
print(target_df.shape)


Y_train = target_df[target_df['use'] == 'training']
Y_val = target_df[target_df['use'] == 'validation']
Y_test = target_df[target_df['use'] == 'test']


Y_train = Y_train.drop(columns = ['use'])
Y_val = Y_val.drop(columns = ['use'])
Y_test = Y_test.drop(columns = ['use'])


print(Y_train.shape)
print(Y_val.shape)
print(Y_test.shape)




print('\n','****************************','\n')



# convert pandas DataFrame into PyTorch Tensor

# train_data = TensorDataset(torch.FloatTensor(X_train.to_numpy()), torch.FloatTensor(Y_train.to_numpy()))
# val_data = TensorDataset(torch.FloatTensor(X_val.to_numpy()), torch.FloatTensor(Y_val.to_numpy()))
# test_data = TensorDataset(torch.FloatTensor(X_test.to_numpy()), torch.FloatTensor(Y_test.to_numpy()))


# train_data = TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train.to_numpy()))
# val_data = TensorDataset(torch.Tensor(X_val.to_numpy()), torch.Tensor(Y_val.to_numpy()))
# test_data = TensorDataset(torch.Tensor(X_test.to_numpy()), torch.Tensor(Y_test.to_numpy()))


train_data = TensorDataset(torch.Tensor(X_train.to_numpy()).to(torch.int64), torch.Tensor(Y_train.to_numpy()).to(torch.int64)) 
val_data = TensorDataset(torch.Tensor(X_val.to_numpy()).to(torch.int64), torch.Tensor(Y_val.to_numpy()).to(torch.int64)) 
test_data = TensorDataset(torch.Tensor(X_test.to_numpy()).to(torch.int64), torch.Tensor(Y_test.to_numpy()).to(torch.int64)) 




# epochs =[10, 25, 50]
epochs =[100, 500,1000]
# epochs =[3]
# batch_size =[32]
batch_size =[32,64]
# learning_rate = [0.01, 0.001,0.05,0.005, 0.0001,0.0005]
# learning_rate = [0.01, 0.001, 0.0001]
# learning_rate = [0.0005, 0.005]
# learning_rate = [0.05]
learning_rate = [0.0001]
# num_layers = [2,3,4]
# num_layers = [2,3]
num_layers = [3]
# hidden_dims = [128,256,512]
hidden_dims = [512]
embedding_dim = 300

device = 'cuda:0'
# device = 'cpu'

criterion = nn.BCELoss()

def acc_(pred, label):

    print('pred size in accuracy > ', pred.size())
    print('label size in accuracy > ', label.size())
    

    pred = torch.round(pred.squeeze())
    label = label.flatten().cpu().numpy()
    
    pred = pred.detach().cpu().numpy()
    print('pred size in accuracy >> ', pred.shape)
    print('label size in accuracy >> ', label.shape)
    acc = np.mean(pred == label)
    print('accuracy is :: ',acc)
    return acc



for b_size in batch_size:
    for dim in hidden_dims:
        for l_rate in learning_rate:
            for layers in num_layers:
              
              for ep in epochs:    
                valid_loss_min =np.Inf
                # epoch_tr_loss, epoch_val_loss = [],[]
                # epoch_tr_acc, epoch_val_acc = [],[]

                # prepare DataLoader
                train_loader = DataLoader(train_data, batch_size = b_size, shuffle = True)
                val_loader = DataLoader(val_data, batch_size = b_size, shuffle = False)
                test_loader = DataLoader(test_data, batch_size = b_size, shuffle = False)

                # def __init__(self, num_layers,hidden_dim, embedding_dim, batch_size, vocab_size_note, vocab_size_gpt4, dropout=0.2)
                # print('dim type is :: ',type(dim), '_',dim)
                # print('layers type is :: ',type(layers),'_',layers)
                model  = RNN_models.Classification_RNN(layers,dim,embedding_dim, b_size, num_unique_words_note, num_unique_words_gpt4)
                optimizer = torch.optim.Adam(model.parameters(), lr = l_rate)
                model.to(device)    
                train_steps = len(train_loader)
                print(model)
                
                
                epoch_tr_loss = []
                epoch_vl_loss = [] 
                epoch_tr_acc = []
                epoch_vl_acc = []
                
                
                folder_path = '/home/ts/Downloads/Glaucoma/' + str(dim) + '_hidden_dims_' + str(l_rate)+ '_lr_' + str(b_size)+'_batch_size_'+ str(layers) + '_layers/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                train_val_loss_path = folder_path + 'Epochs_' + str(ep) + '_epochs_train_val.txt'
                
                f = open(train_val_loss_path, 'a')

                time_now = time.time()
                for e in range(ep):
                        iter_count = 0
                        f.write('\n')
                        f.write('Epoch '+str(e+1)+' ::')
                        f.write('\n')
                        f.write('\n')
                        f.write('=================')
                        f.write('\n')
                        train_losses = []
                        tr_preds = []
                        tr_labels = []
                        model.train()
                        for i, (inputs,labels) in enumerate(train_loader):    
                            # print('i : ', i)
                            iter_count += 1
                            inputs, labels  = inputs.to(device), labels.to(device)
                            # print(25*'**')
                            # print('inputs >> ',inputs.size())
                            # print('labels >> ',labels.size())
                            
                            
                            h = model.init_hidden(inputs.size(0),device)
                            train_h = tuple([each.data for each in h])
                            
                            model.zero_grad()
                            output = model(inputs,train_h)
                            tr_preds.append(output)
                            tr_labels.append(labels)
                            # print('output  size :: ',output.size())
                            # print('hidden  size :: ',hidden.size())
                            # calculate the loss and perform backprop
                            # loss = criterion(output.squeeze(), labels.float())
                            loss = criterion(output, labels.float())
                            loss.backward()
                            train_losses.append(loss.item())
                           

                            #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                            # nn.utils.clip_grad_norm_(model.parameters(), clip)
                            
                            optimizer.step()
                            if (i + 1) % 10 == 0:
                            
                                print("\tepoch: {0}, iters: {1} | loss: {2:.7f}".format(e + 1, i + 1, loss.item()))
                                f.write("\tepoch: {0}, iters: {1} | loss: {2:.7f}".format(e + 1, i + 1, loss.item()))
                                # f.write("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, e + 1, loss.item()))
                                speed = (time.time() - time_now) / iter_count
                                # left_time = speed * ((self.train_epochs - epoch) * train_steps - i)
                                # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

                                # f.write('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                                f.write('\n')
                                iter_count = 0
                                time_now = time.time()

                        tr_preds = torch.cat(tr_preds,0)
                        tr_labels = torch.cat(tr_labels,0) 
                        # print('tr_preds size :: ', tr_preds.size())
                        # print('tr_labels size :: ', tr_labels.size())
                        tr_acc = acc_(tr_preds, tr_labels)  
                        # print('training accuracy is : ', tr_acc)
                        # print(f'train_accuracy : {tr_acc*100}')
                        # print(tess)   
                        
                        
                        model.eval()
                        val_losses = []
                        val_preds = []
                        val_labels = []
                        with torch.no_grad():
                         for inputs, labels in val_loader:
                            
                            h = model.init_hidden(inputs.size(0),device)
                            val_h = tuple([each.data for each in h])
                            inputs, labels = inputs.to(device), labels.to(device)
                            output = model(inputs, val_h)
                            val_preds.append(output)
                            val_labels.append(labels)                            
                            val_loss = criterion(output, labels.float())
                            val_losses.append(val_loss.item())
            
            
                        val_preds = torch.cat(val_preds,0)
                        val_labels = torch.cat(val_labels,0) 
                        # print('val_preds size :: ', val_preds.size())
                        # print('val_labels size :: ', val_labels.size())
                        val_acc = acc_(val_preds, val_labels)  
                        # print('validation accuracy is : ', val_acc)
                        # print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
                        print(f'validation_accuracy : {val_acc*100}')    
                       
            
                        tr_loss = np.mean(train_losses)
                        val_loss = np.mean(val_losses)
                        
                        epoch_tr_loss.append(tr_loss)
                        epoch_vl_loss.append(val_loss)
                        
                        epoch_tr_acc.append(tr_acc)
                        epoch_vl_acc.append(val_acc)
                        # print(f'Epoch {e+1}')
                        # print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
                        # print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
                        
                        
                        print("Epoch: {0} Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Train Acc: {4:.7f} Val Acc: {5:.7f}".format(
                                            e + 1, train_steps, tr_loss, val_loss, tr_acc,val_acc))
                        f.write("Epoch: {0} Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Train Acc: {4:.7f} Val Acc: {5:.7f} ".format(
                                            e + 1,  train_steps, tr_loss, val_loss, tr_acc,val_acc))
                        
                        
                        
                        # print(tess)
                        # folder_path = '/home/ts/Downloads/Glaucoma/' + str(dim) + '_hidden_dims_' + str(l_rate)+ '_lr_' + str(b_size)+'_batch_size_'+ str(layers) + '_layers/'

                        file_name =  'Epochs_' + str(ep) + '_state_dict.pt'
                        best_model_path = folder_path + '/'+ file_name
                        
                        # if not os.path.exists(folder_path):
                            # os.makedirs(folder_path)
                        
                        
                        if val_acc <= valid_loss_min:

                            torch.save(model.state_dict(), best_model_path)
                            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,val_acc))
                            valid_loss_min = val_acc
                        print(25*'==')
                    
                        # best_model_path = folder_path +'/checkpoint_state_dict.pth'
                       
                        model.load_state_dict(torch.load(best_model_path))
                        # print(tess)



                
                
                # [plot the accuracy and the loss curves]
                
                fig = plt.figure(figsize = (20, 6))
                plt.subplot(1, 2, 1)
                plt.plot(epoch_tr_acc, label='Train Acc')
                plt.plot(epoch_vl_acc, label='Validation Acc')
                plt.title("Accuracy")
                plt.legend()
                plt.grid()
    
                plt.subplot(1, 2, 2)
                plt.plot(epoch_tr_loss, label='Train loss')
                plt.plot(epoch_vl_loss, label='Validation loss')
                plt.title("Loss")
                plt.legend()
                plt.grid()

                # plt.show()
                f_name_tr_vl_acc_loss_plt = folder_path + 'Epochs_' + str(ep) + '_tr_vl_acc_loss_plt.png'
                 

                # f_name_tr_vl_loss_plt = folder_path +  'tr_vl_loss_plt.png'
                 



                plt.savefig(f_name_tr_vl_acc_loss_plt)      # save train val accuracy and train val loss plots
                
                # Define evaluate_model function
                test_losses = []
                test_preds = []
                test_labels = []
                model.eval()
                for inputs, labels in test_loader:
                    

                    h = model.init_hidden(inputs.size(0),device)
                    test_h = tuple([each.data for each in h])

                    inputs, labels = inputs.to(device), labels.to(device)

                    output = model(inputs, test_h)
                    # val_loss = criterion(output.squeeze(), labels.float())
                    test_loss = criterion(output, labels.float())

    
                    test_preds.append(output)
                    test_labels.append(labels)  
                    test_losses.append(test_loss.item())
                    
              
              
                test_preds = torch.cat(test_preds,0)
                test_labels = torch.cat(test_labels,0)
                
                test_loss = np.mean(test_losses)
                test_acc = acc_(test_preds, test_labels)  
                print(f'test_accuracy : {test_acc*100}')
                
                
                test_path = folder_path + 'Epochs_' + str(ep) + '_test'+ '.txt'
                
                f = open(test_path, 'a')
                f.write('\n')         
                f.write("Test Loss: {0:.7f} Test Acc: {1:.7f} ".format(test_loss, test_acc))                        
                f.write('\n')
                f.close()
               
             