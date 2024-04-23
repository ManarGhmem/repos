# import random
# import json
# import torch
# from chatbot.model import NeuralNet
# from chatbot.nltk_utils import bag_of_words, tokenize
# from nltk.stem.porter import PorterStemmer
# from chatbot.model import Transformer
# from chatbot.nltk_utils import pad_sequence
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# import os

# # Obtenez le chemin absolu du répertoire actuel
# current_directory = os.path.dirname(__file__)

# # Chemin relatif vers le fichier intents.json
# file_path = os.path.join(current_directory, 'intents.json')

# # Ouvrir et charger le fichier JSON
# with open(file_path, 'r') as f:
#     intents = json.load(f)

# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# model = Transformer(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "Sam"

# def get_response(msg):
#     sentence = tokenize(msg)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 return random.choice(intent['responses'])
    
#     return "I do not understand..."


# if __name__ == "__main__":
#     print("Let's chat! (type 'quit' to exit)")
#     while True:
#         # sentence = "do you use credit cards?"
#         sentence = input("You: ")
#         if sentence == "quit":
#             break

#         resp = get_response(sentence)
#         print(resp)







import random
import json
import torch
from chatbot.model import Transformer
from chatbot.nltk_utils import bag_of_words, tokenize
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Obtenez le chemin absolu du répertoire actuel
current_directory = os.path.dirname(__file__)

# Chemin relatif vers le fichier intents.json
file_path = os.path.join(current_directory, 'intents.json')

# Ouvrir et charger le fichier JSON
with open(file_path, 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = Transformer(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)



# import random
# import json
# import torch
# import nltk
# import numpy as np
# from nltk.stem.porter import PorterStemmer
# from chatbot.model import Transformer
# from chatbot.nltk_utils import pad_sequence
# import os

# # Assurez-vous que le téléchargement des données nltk est fait une seule fois
# nltk.download('punkt', quiet=True)

# # Charger les données nltk nécessaires
# stemmer = PorterStemmer()

# # Obtenez le chemin absolu du répertoire actuel
# current_directory = os.path.dirname(__file__)

# # Chemin relatif vers le fichier intents.json
# file_path = os.path.join(current_directory, 'intents.json')

# # Ouvrir et charger le fichier JSON
# with open(file_path, 'r') as f:
#     intents = json.load(f)

# # Charger les données du modèle
# SAVE_FILE = "saved_data.pth"
# save_data = torch.load(SAVE_FILE)

# # Extraire les données nécessaires
# model_state_dict = save_data['model_state_dict']
# src_tokenizer = save_data['src_tokenizer']
# trg_tokenizer = save_data['trg_tokenizer']
# SRC_MAXLEN = save_data['SRC_MAXLEN']
# TRG_MAXLEN = save_data['TRG_MAXLEN']

# # Définir le modèle
# model = Transformer(
#     NUM_LAYERS ,
#     EMBEDDING_DIM ,
#     NUM_HEADS ,
#     FC_DIM ,
#     SRC_VOCAB_SIZE ,
#     TRG_VOCAB_SIZE ,
#     SRC_MAXLEN ,
#     TRG_MAXLEN ,
#     DROPOUT_RATE
# ).to(device)
# model.load_state_dict(model_state_dict)
# model.eval()

# bot_name = "Sam"

# def tokenize(sentence):
#     return nltk.word_tokenize(sentence)

# def stem(word):
#     return stemmer.stem(word.lower())

# def get_response(msg):
#     # Tokenisation du message
#     sentence = tokenize(msg)
#     # Padding de la séquence
#     padded_seq = pad_sequence([sentence], SRC_MAXLEN)
#     # Vectorisation de la séquence
#     X = torch.tensor(padded_seq, dtype=torch.int64).to(device)

#     # Inférence
#     output = evaluate(X)
#     # Conversion de l'index en séquence de mots
#     predicted_sentence = ' '.join([trg_tokenizer.index_word[idx] for idx in output.cpu().numpy() if idx != 0 and idx != 2])

#     return predicted_sentence

# if __name__ == "__main__":
#     print("Let's chat! (type 'quit' to exit)")
#     while True:
#         sentence = input("You: ")
#         if sentence == "quit":
#             break

#         resp = get_response(sentence)
#         print(resp)









# import os
# import json
# import torch
# import nltk
# import numpy as np
# from nltk.stem.porter import PorterStemmer
# from chatbot.model import Transformer
# from chatbot.nltk_utils import pad_sequence
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # Assurez-vous que le téléchargement des données nltk est fait une seule fois
# nltk.download('punkt', quiet=True)

# stemmer = PorterStemmer()

# def tokenize(sentence):
#     return nltk.word_tokenize(sentence)

# def stem(word):
#     return stemmer.stem(word.lower())

# def bag_of_words(tokenized_sentence, all_words):
#     sentence_words = [stem(word) for word in tokenized_sentence]
#     # Initialize bag with 0 for each word
#     bag = np.zeros(len(all_words), dtype=np.float32)
#     for idx, w in enumerate(all_words):
#         if w in sentence_words:
#             bag[idx] = 1

#     return bag
# # Obtenez le chemin absolu du répertoire actuel
# current_directory = os.path.dirname(__file__)

# # Chemin relatif vers le fichier intents.json
# file_path = os.path.join(current_directory, 'intents.json')

# # Ouvrir et charger le fichier JSON
# with open(file_path, 'r') as f:
#     intents = json.load(f)


# # Obtenez le chemin absolu du répertoire actuel
# current_directory = os.path.dirname(__file__)

# # Chemin relatif vers le fichier saved_data.pth
# save_path = os.path.join(current_directory, 'saved_data.pth')

# # Charger les données du modèle
# save_data = torch.load(save_path)


# # Extraire les données nécessaires
# model_state_dict = save_data['model_state_dict']
# src_tokenizer = save_data['src_tokenizer']
# trg_tokenizer = save_data['trg_tokenizer']
# SRC_MAXLEN = save_data['SRC_MAXLEN']
# TRG_MAXLEN = save_data['TRG_MAXLEN']

# # Définir le modèle
# model = Transformer(
#     num_layers=4,
#     embedding_dim=256,
#     num_heads=8,
#     fc_dim=512,
#     src_vocab_size=len(src_tokenizer.word_index),
#     trg_vocab_size=len(trg_tokenizer.word_index),
#     src_max_length=SRC_MAXLEN,
#     trg_max_length=TRG_MAXLEN,
#     dropout_rate=0.3
# ).to(DEVICE)
# model.load_state_dict(model_state_dict)
# model.eval()

# bot_name = "Sam"

# def tokenize(sentence):
#     return nltk.word_tokenize(sentence)

# def stem(word):
#     return stemmer.stem(word.lower())
# def get_response(msg):
#     # Tokenisation du message
#     sentence = tokenize(msg)
#     # Padding de la séquence
#     padded_seq = pad_sequence([sentence], SRC_MAXLEN)
#     print("Padded Sequence:", padded_seq)  
#     # Conversion de la séquence en un tenseur PyTorch
#     X = torch.tensor(padded_seq, dtype=torch.int64).to(DEVICE)

#     # Supposons que votre modèle ait une méthode predict pour obtenir les prédictions
#     with torch.no_grad():
#         output = model(X)
    
#     # Supposons que vous récupériez l'index du mot prédit avec la plus haute probabilité
#     predicted_index = torch.argmax(output, dim=2).item()

#     # Supposons que vous utilisez un index pour récupérer le mot prédit à partir du tokenizer
#     predicted_word = trg_tokenizer.index_word[predicted_index]

#     return predicted_word
