# import json
# import os
# import numpy as np 
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from nltk_utils import tokenize, stem, bag_of_words
# from model import NeuralNet

# # Obtenez le chemin absolu du répertoire actuel
# current_directory = os.path.dirname(__file__)

# # Chemin relatif vers le fichier intents.json
# file_path = os.path.join(current_directory, 'intents.json')

# # Ouvrir et charger le fichier JSON
# with open(file_path, 'r', encoding='utf-8') as f:
#     intents = json.load(f)

# all_words = []
# tags = []
# xy = []
# # loop through each sentence in our intents patterns
# for intent in intents['intents']:
#     tag = intent['tag']
#     # add to tag list
#     tags.append(tag)
#     for pattern in intent['patterns']:
#         # tokenize each word in the sentence
#         w = tokenize(pattern)
#         # add to our words list
#         all_words.extend(w)
#         # add to xy pair
#         xy.append((w, tag))

# # stem and lower each word
# ignore_words = ['?', '.', '!']
# all_words = [stem(w) for w in all_words if w not in ignore_words]
# # remove duplicates and sort
# all_words = sorted(set(all_words))
# tags = sorted(set(tags))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# # create training data
# X_train = []
# y_train = []
# for (pattern_sentence, tag) in xy:
#     # X: bag of words for each pattern_sentence
#     bag = bag_of_words(pattern_sentence, all_words)
#     X_train.append(bag)
#     # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
#     label = tags.index(tag)
#     y_train.append(label)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# # Hyper-parameters 
# num_epochs = 1000
# batch_size = 8
# learning_rate = 0.001
# input_size = len(X_train[0])
# hidden_size = 8
# output_size = len(tags)
# print(input_size, output_size)

# class ChatDataset(Dataset):

#     def __init__(self):
#         self.n_samples = len(X_train)
#         self.x_data = X_train
#         self.y_data = y_train

#     # support indexing such that dataset[i] can be used to get i-th sample
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     # we can call len(dataset) to return the size
#     def __len__(self):
#         return self.n_samples

# dataset = ChatDataset()
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           num_workers=0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = NeuralNet(input_size, hidden_size, output_size).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
# for epoch in range(num_epochs):
#     for (words, labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)
        
#         # Forward pass
#         outputs = model(words)
#         # if y would be one-hot, we must apply
#         # labels = torch.max(labels, 1)[1]
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#     if (epoch+1) % 100 == 0:
#         print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# print(f'final loss: {loss.item():.4f}')

# data = {
# "model_state": model.state_dict(),
# "input_size": input_size,
# "hidden_size": hidden_size,
# "output_size": output_size,
# "all_words": all_words,
# "tags": tags
# }

# FILE = "data.pth"
# torch.save(data, FILE)

# print(f'training complete. file saved to {FILE}')





import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from collections import Counter
from string import punctuation
from wordcloud import WordCloud
from model import Transformer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch import nn
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader , TensorDataset
from torchinfo import summary
from torchmetrics.text import BLEUScore
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} DEVICE")
import pandas as pd
import json
import os 

# Obtenez le chemin absolu du répertoire actuel
current_directory = os.path.dirname(__file__)

# Chemin relatif vers le fichier intents.json
file_path = os.path.join(current_directory, 'intents.json')

# Load the JSON data
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


pattern_response_pairs = []

# Extract patterns and responses for each intent
for intent in data['intents']:
    patterns = intent['patterns']
    responses = intent['responses']

    # Create pairs of patterns and responses
    for pattern in patterns:
        for response in responses:
            pattern_response_pairs.append({'Pattern': pattern, 'Response': response})

# Create a DataFrame
df = pd.DataFrame(pattern_response_pairs)

# Display the DataFrame
df['pattern_len'] = [len(text.split()) for text in df.Pattern]
df['response_len'] = [len(text.split()) for text in df.Response]
# print(df.head(5))
def src_preprocessing(data , col) :
    data[col] = data[col].astype(str)
    data[col] = data[col].apply(lambda x: x.lower())
    data[col] = data[col].apply(lambda x: re.sub("[^A-Za-z\s]","",x))
    data[col] = data[col].apply(lambda x: x.replace("\s+"," "))
    data[col] = data[col].apply(lambda x: " ".join([word for word in x.split()]))
    return data

def trg_preprocessing(data , col) :
    data[col] = data[col].astype(str)
    data[col] = data[col].apply(lambda x : x.lower())
    data[col] = data[col].apply(lambda x: re.sub(r'\d','',x))
    data[col] = data[col].apply(lambda x: re.sub(r'\s+',' ',x))
    data[col] = data[col].apply(lambda x: re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,।]", "", x))
    data[col] = data[col].apply(lambda x: x.strip())
    data[col] = "<sos> " + data[col] + " <eos>"
    return data

df = src_preprocessing(df , 'Pattern')
df = trg_preprocessing(df , 'Response')
# print(df)
# df = df[~(df['pattern_len'] < 5) & ~(df['pattern_len'] > 19)]
# df = df[~(df['response_len'] < 5) & ~(df['response_len'] > 19)]
print(np.min(df['pattern_len']) , np.min(df['response_len']))
print(np.max(df['pattern_len']) , np.max(df['response_len']))
SRC_MAXLEN = np.max(df['pattern_len'])
TRG_MAXLEN = np.max(df['response_len'])
def Vectorization(col , MAXLEN) :
    sents = df[col].tolist()

    # build vocabulary
    corpus = [word for text in df[col] for word in text.split()]
    vocab_size = len(Counter(corpus))

    tokenizer = Tokenizer(num_words=vocab_size , oov_token = "<OOV>" ,
                          filters='!#$%&()*+,-/:;<=>@«»""[\\]^_`{|}~\t\n'
                         )

    tokenizer.fit_on_texts(sents)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    vocab_to_idx = tokenizer.word_index
    idx_to_vocab = tokenizer.index_word

    # Text Vectorization
    seqs = tokenizer.texts_to_sequences(sents)

    pad_seqs = pad_sequences(seqs , maxlen = MAXLEN , padding = 'post')

    return pad_seqs , tokenizer

pattern_seqs , src_tokenizer = Vectorization('Pattern' , SRC_MAXLEN)
response_seqs , trg_tokenizer = Vectorization('Response' , TRG_MAXLEN)
print(pattern_seqs.shape)
print(response_seqs.shape)
print(f"The size of the source vocab size : {len(src_tokenizer.word_index)}\n")
print(f"The size of the target vocab size : {len(trg_tokenizer.word_index)}\n")
trg_sent = ' '.join([trg_tokenizer.index_word[idx] for idx in response_seqs[15] if idx != 0])
print(f"{response_seqs[15]} \n\n {trg_sent}")
BATCH_SIZE = 1
ds = TensorDataset(torch.LongTensor(pattern_seqs) , torch.LongTensor(response_seqs))
torch.manual_seed(45)
ds_dataloader = DataLoader(
    dataset = ds ,
    batch_size = BATCH_SIZE ,
    shuffle = True ,
    num_workers = 0 ,
    pin_memory = True
)
print(f"the size of the dataloader {len(ds_dataloader)} batches of {BATCH_SIZE}")

# set hyperparameters
EPOCHS = 100
LR = 1e-3 # 0.0003
EMBEDDING_DIM = 256
FC_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT_RATE = 0.3
SRC_VOCAB_SIZE = len(src_tokenizer.word_index) # 1978
TRG_VOCAB_SIZE = len(trg_tokenizer.word_index) # 1987
SRC_MAXLEN = np.max(df['pattern_len'])
TRG_MAXLEN = np.max(df['response_len'])
print(SRC_VOCAB_SIZE," ",TRG_VOCAB_SIZE)
# affect model into device
model = Transformer(
    NUM_LAYERS ,
    EMBEDDING_DIM ,
    NUM_HEADS ,
    FC_DIM ,
    SRC_VOCAB_SIZE ,
    TRG_VOCAB_SIZE ,
    SRC_MAXLEN ,
    TRG_MAXLEN ,
    DROPOUT_RATE
).to(DEVICE)
temp_src = torch.randint(low=0, high=91, size=(BATCH_SIZE , SRC_MAXLEN), dtype=torch.int64).to(DEVICE)
temp_trg = torch.randint(low=0, high=409, size=(BATCH_SIZE , TRG_MAXLEN), dtype=torch.int64).to(DEVICE)

temp_trg_out = model(temp_src, temp_trg)
print(temp_trg_out.shape)
print(summary(model , input_data = [temp_src , temp_trg]))
src_len = temp_src.shape[1]  # Get the length of source sequences
trg_len = temp_trg.shape[1]  # Get the length of target sequences

#train
criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.word_index['<pad>'])
optimizer = Adam(model.parameters(), lr=LR)
def train_step(src , trg) :
    decoder_input = trg[: , :-1]
    trg_reals = trg[: , 1:].reshape(-1)

    preds = model(src , decoder_input)

    preds = preds.reshape(-1 , preds.shape[2])

    optimizer.zero_grad()

    loss = criterion(preds , trg_reals)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters() , max_norm=1) # avoid exploding gradient issues

    optimizer.step()

    return loss
train_losses = []
for epoch in tqdm(range(EPOCHS)) :
    epoch_loss = 0

    model.train()
    for src , trg in ds_dataloader :
        src , trg = src.to(DEVICE) , trg.to(DEVICE)
        loss = train_step(src , trg)

        epoch_loss += loss

    train_losses.append((epoch_loss / len(ds_dataloader)).cpu().detach().numpy())
    if (epoch + 1) % 20 == 0 :
            print(f"\n[Epoch :  {epoch+1}/{EPOCHS}] [Train Loss : {train_losses[-1]:0.2f}]\n")
plt.plot(train_losses , label='train')
plt.title('Training loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
def evaluate(sent) :
    sentence = src_tokenizer.texts_to_sequences([sent])
    sentence = pad_sequences(sentence , maxlen = SRC_MAXLEN , padding = 'post')

    src_input = torch.tensor(np.array(sentence) , dtype = torch.int64)

    decoder_input = trg_tokenizer.texts_to_sequences(['sos'])
    decoder_input = torch.tensor(np.array(decoder_input) , dtype = torch.int64)

    src_input , decoder_input = src_input.to(DEVICE) , decoder_input.to(DEVICE)

    for i in range(TRG_MAXLEN) :
        preds = model(src_input , decoder_input)

        preds = preds[: , -1: , :] # (batch_size, 1, vocab_size)

        predicted_id = torch.argmax(preds, dim=-1)

        if predicted_id.item() == trg_tokenizer.word_index['eos'] :
            return decoder_input.squeeze(0)

        decoder_input = torch.cat([decoder_input , predicted_id] , dim = 1)

    return decoder_input.squeeze(0)
test_sample = df.sample(10)
x_test = test_sample['Pattern'].tolist()
y_test = test_sample['Response'].tolist()

for idx, (src_sent, trg_sent) in enumerate(zip(x_test[-10:], y_test[-10:])):
    result = evaluate(src_sent)
    pred_sent = ' '.join([trg_tokenizer.index_word[idx] for idx in result.cpu().numpy() if idx != 0 and idx != 2])
    print(f"Input sentence {idx+1} : {src_sent}")
    print(f"Actual correction {idx+1} : {trg_sent}")
    print(f"Predicted correction {idx+1} : {pred_sent}\n")

# # Sauvegarde du modèle
# data = {
#     "model_state": model.state_dict(),
#     "input_size": input_size,
#     "hidden_size": hidden_size,
#     "output_size": output_size,
#     "all_words": all_words,
#     "tags": tags
# }

# FILE = "data.pth"
# torch.save(data, FILE)

# print(f'Training complete. File saved to {FILE}')
# Sauvegarde du modèle et des autres informations nécessaires
save_data = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'src_tokenizer': src_tokenizer,
    'trg_tokenizer': trg_tokenizer,
    'SRC_MAXLEN': SRC_MAXLEN,
    'TRG_MAXLEN': TRG_MAXLEN,
    'train_losses': train_losses,
    # Ajoutez d'autres informations si nécessaire
}

# Définissez le chemin où vous souhaitez enregistrer les données
save_path = os.path.join(current_directory, 'saved_data.pth')

# Enregistrez les données
torch.save(save_data, save_path)
print(f"Données d'entraînement enregistrées à {save_path}")

