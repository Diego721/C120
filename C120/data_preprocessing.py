# Bibliotecas de preprocesamiento de datos de texto
import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import json
import pickle
import numpy as np
import random

words=[] # Lista de palabras raíz únicas en los datos
classes = [] # Lista de etiquetas únicas en los datos
# Lista de pares de la forma: (['palabras', 'de', 'la', 'oración'], 'etiquetas')
pattern_word_tags_list = [] 
ignore_words = ['?', '!',',','.', "'s", "'m"]

train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words
 
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in intents['intents']:

        # Agregar todos los patrones y etiquetas a la lista
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                        
            pattern_word_tags_list.append((pattern_word, intent['tag']))
              
        # Agregar todas las etiquetas a la lista "classes"
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
    stem_words = get_stem_words(words, ignore_words) 
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    return stem_words, classes, pattern_word_tags_list


# Conjunto de datos de entrenamiento: 
# Texto de entrada ----> como una bolsa de palabras
# Etiquetas -----------> como etiqueta

def bag_of_words_encoding(stem_words, pattern_word_tags_list):  
    bag = []
    for word_tags in pattern_word_tags_list:

        pattern_words = word_tags[0] 
        bag_of_words = []
        stem_pattern_words= get_stem_words(pattern_words, ignore_words)
        for word in stem_words:            
            if word in stem_pattern_words:              
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        bag.append(bag_of_words)
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    labels = []
    for word_tags in pattern_word_tags_list:

        labels_encoding = list([0]*len(classes)) 
        tag = word_tags[1]
        tag_index = classes.index(tag)
        labels_encoding[tag_index] = 1
        labels.append(labels_encoding)
    return np.array(labels)

def preprocess_train_data():
  
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, 
                                            pattern_word_tags_list, ignore_words)
    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(tag_classes, open('classes.pkl','wb'))
    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

# preprocess_train_data()


