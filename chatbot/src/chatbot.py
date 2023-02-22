import random
import json
import pickle

import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from keras.models import load_model


lemmatizer = WordNetLemmatizer()

with open('../models/training_data.py', 'rb') as file:
    data = pickle.load(file)

with open('intents.json', 'r') as file:
    intents = json.load(file)


words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

model = load_model('models/model.tflearn')

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # lemmatize each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    # tokenizing the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    # filter out predictions below a threshold
    p = bag_of_words(sentence)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# create a chatbot that will loop until the user exits
while True:
    message = input("You: ")
    if message == "quit":
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("ChatBot: " + res)
