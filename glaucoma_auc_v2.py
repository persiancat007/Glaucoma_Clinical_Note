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
from sklearn.metrics import roc_auc_score


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


train_data = TensorDataset(torch.Tensor(X_train.to_numpy()).to(torch.int64), torch.Tensor(Y_train.to_numpy()).to(torch.int64)) 
val_data = TensorDataset(torch.Tensor(X_val.to_numpy()).to(torch.int64), torch.Tensor(Y_val.to_numpy()).to(torch.int64)) 
test_data = TensorDataset(torch.Tensor(X_test.to_numpy()).to(torch.int64), torch.Tensor(Y_test.to_numpy()).to(torch.int64)) 


device = 'cuda:0'


print('\n','****************************','\n')

model_path = '/home/ts/Downloads/Glaucoma/256_hidden_dims_0.0001_lr_32_batch_size_2_layers/' +'Epochs_500_state_dict.pt'
model  = RNN_models.Classification_RNN(2,256,300, 32, num_unique_words_note, num_unique_words_gpt4)
b_size = 32        

checkpoint = torch.load(model_path, weights_only=True)
model.load_state_dict(checkpoint)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.load_state_dict(torch.load(model_path, weights_only=True))

model.eval()
model.to(device)    



val_loader = DataLoader(val_data, batch_size = b_size, shuffle = False)
test_loader = DataLoader(test_data, batch_size = b_size, shuffle = False)




val_losses = []
val_preds = []
val_labels = []
for inputs, labels in val_loader:
    
    h = model.init_hidden(inputs.size(0),device)
    val_h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs, val_h)
    val_preds.append(output)
    val_labels.append(labels)    


val_preds = torch.cat(val_preds,0)
val_labels = torch.cat(val_labels,0) 


test_losses = []
test_preds = []
test_labels = []
for inputs, labels in test_loader:
                    

    h = model.init_hidden(inputs.size(0),device)
    test_h = tuple([each.data for each in h])

    inputs, labels = inputs.to(device), labels.to(device)

    output = model(inputs, test_h)
    # val_loss = criterion(output.squeeze(), labels.float())
    # test_loss = criterion(output, labels.float())


    test_preds.append(output)
    test_labels.append(labels)  



# print(len(test_preds))
# print(len(test_labels))
test_preds = torch.cat(test_preds,0)
test_labels = torch.cat(test_labels,0)


df_one_hot_df_val_pred = one_hot_df[one_hot_df['use'] == 'validation']
df_one_hot_df_test_pred = one_hot_df[one_hot_df['use'] == 'test']




df_one_hot_df_val_pred = df_one_hot_df_val_pred.drop(columns = ['use'])
df_one_hot_df_test_pred = df_one_hot_df_test_pred.drop(columns = ['use'])

df_one_hot_df_val_race = df_one_hot_df_val_pred.loc[:, ['race_asian', 'race_black', 'race_white']] 
df_one_hot_df_test_race = df_one_hot_df_test_pred.loc[:, ['race_asian', 'race_black', 'race_white']] 



print('\n','****************************','\n')
print(one_hot_df.shape)
print(one_hot_df.columns)
print('\n','****************************','\n')
print(df_one_hot_df_test_pred.columns)
print(df_one_hot_df_test_pred.shape)
print('\n','****************************','\n')
print(df_one_hot_df_val_pred.columns)
print(df_one_hot_df_val_pred.shape)
print('\n','****************************','\n')
print(df_one_hot_df_val_race.columns)
print(df_one_hot_df_val_race.shape)
print('\n','****************************','\n')
print(df_one_hot_df_test_race.columns)
print(df_one_hot_df_test_race.shape)
print('\n','****************************','\n')

# print(df_one_hot.shape)
# print(df_one_hot_df.shape)


print('test_preds size :: ', test_preds.size())
print('test_labels size :: ', test_labels.size())

print('val_preds size :: ', val_preds.size())
print('val_labels size :: ', val_labels.size())


# test_preds = torch.round(test_preds.squeeze())


test_labels = test_labels.flatten().cpu().numpy()    
test_preds = test_preds.squeeze().detach().cpu().numpy()

print(test_labels.shape)
print(test_preds.shape)

val_labels = val_labels.flatten().cpu().numpy()    
val_preds = val_preds.squeeze().detach().cpu().numpy()


print(val_labels.shape)
print(val_preds.shape)



# tmp_test = (test_preds == test_labels)

# print(type(tmp_test))
# print(tmp_test)





df_one_hot_df_test_race['glaucoma_true'] = test_labels.tolist()
df_one_hot_df_test_race['glaucoma_pred'] = test_preds.tolist()


df_one_hot_df_val_race['glaucoma_true'] = val_labels.tolist()
df_one_hot_df_val_race['glaucoma_pred'] = val_preds.tolist()


print('\n','****************************','\n')
print(df_one_hot_df_val_race.columns)
print(df_one_hot_df_val_race.shape)
print('\n','****************************','\n')
print(df_one_hot_df_test_race.columns)
print(df_one_hot_df_test_race.shape)
print('\n','****************************','\n')

# appply whole AUC here

val_true = df_one_hot_df_val_race['glaucoma_true']
val_pred = df_one_hot_df_val_race['glaucoma_pred']


test_true = df_one_hot_df_test_race['glaucoma_true']
test_pred = df_one_hot_df_test_race['glaucoma_pred']



val_auc = np.round(roc_auc_score(val_true, val_pred),3)
test_auc = np.round(roc_auc_score(test_true, test_pred),3)

print('AUC for overall val data is {}'.format(val_auc))
print('AUC for overall test data is {}'.format(test_auc))


print('\n','++++++++++++++++++++++++','\n')

val_asian = df_one_hot_df_val_race[df_one_hot_df_val_race['race_asian'] == 1]
val_black = df_one_hot_df_val_race[df_one_hot_df_val_race['race_black'] == 1]
val_white = df_one_hot_df_val_race[df_one_hot_df_val_race['race_white'] == 1]


print(val_asian.shape)
print(val_black.shape)
print(val_white.shape)


val_asian.to_csv('./glaucoma_val_asian_2.csv')
val_black.to_csv('./glaucoma_val_black_2.csv')
val_white.to_csv('./glaucoma_val_white_2.csv')
print('\n','*************************','\n')

test_asian = df_one_hot_df_test_race[df_one_hot_df_test_race['race_asian'] == 1]
test_black = df_one_hot_df_test_race[df_one_hot_df_test_race['race_black'] == 1]
test_white = df_one_hot_df_test_race[df_one_hot_df_test_race['race_white'] == 1]

print(test_asian.shape)
print(test_black.shape)
print(test_white.shape)


test_asian.to_csv('./glaucoma_test_asian_2.csv')
test_black.to_csv('./glaucoma_test_black_2.csv')
test_white.to_csv('./glaucoma_test_white_2.csv')

val_asian = val_asian.drop(columns = ['race_black', 'race_white'])
test_asian = test_asian.drop(columns = ['race_black', 'race_white'])

val_black = val_black.drop(columns = ['race_asian', 'race_white'])
test_black = test_black.drop(columns = ['race_asian', 'race_white'])

val_white = val_white.drop(columns = ['race_black', 'race_asian'])
test_white = test_white.drop(columns = ['race_black', 'race_asian'])

# Apply asian black white individual AUC s here

# test_asian = df_one_hot_df_test_race[df_one_hot_df_test_race['race_asian'] == '1']
# test_black = df_one_hot_df_test_race[df_one_hot_df_test_race['race_black'] == '1'] 
# test_white = df_one_hot_df_test_race[df_one_hot_df_test_race['race_white'] == '1']




print(val_asian.shape)
print(val_black.shape)
print(val_white.shape)

val_asian_true = val_asian['glaucoma_true']
val_asian_pred = val_asian['glaucoma_pred']


val_black_true = val_black['glaucoma_true']
val_black_pred = val_black['glaucoma_pred']


val_white_true = val_white['glaucoma_true']
val_white_pred = val_white['glaucoma_pred']

val_asian_auc = np.round(roc_auc_score(val_asian_true, val_asian_pred),3)
val_black_auc = np.round(roc_auc_score(val_black_true, val_black_pred),3)
val_white_auc = np.round(roc_auc_score(val_white_true, val_white_pred),3)

print('AUC for val asian data is {}'.format(val_asian_auc))
print('AUC for val black data is {}'.format(val_black_auc))
print('AUC for val white data is {}'.format(val_white_auc))

print('\n','########################','\n')

print(test_asian.shape)
print(test_black.shape)
print(test_white.shape)


test_asian_true = test_asian['glaucoma_true']
test_asian_pred = test_asian['glaucoma_pred']


test_black_true = test_black['glaucoma_true']
test_black_pred = test_black['glaucoma_pred']


test_white_true = test_white['glaucoma_true']
test_white_pred = test_white['glaucoma_pred']

test_asian_auc = np.round(roc_auc_score(test_asian_true, test_asian_pred),3)
test_black_auc = np.round(roc_auc_score(test_black_true, test_black_pred),3)
test_white_auc = np.round(roc_auc_score(test_white_true, test_white_pred),3)

print('AUC for test asian data is {}'.format(test_asian_auc))
print('AUC for test black data is {}'.format(test_black_auc))
print('AUC for test white data is {}'.format(test_white_auc))

print('\n','************************','\n')