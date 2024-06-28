import nltk # natural lang tool kit
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle
import numpy as np

words=[]
classes=[] #[greeting]
word_tags_list=[]
ignore_words=['?','!','.',',',"'s","'m"]
train_data_file =open("intents.json").read()
intents = json.loads(train_data_file)

def get_stem_words(words,ignore_words): # how are you? =how be you
    stem_words=[]
    for word in words:
        if word not in ignore_words:
            w=stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words
for intent in intents['intents']:
    for pattern in intent['patterns']: # how are you?
        pattern_word = nltk.word_tokenize(pattern) # ["how","are","you"]
        words.extend(pattern_word)
        word_tags_list.append((pattern_word,intent['tag'])) #[how are you ,greeting]
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
        stem_words = get_stem_words(words,ignore_words)

print(stem_words)
print("----------------------------")
print(word_tags_list[0])
print("-----------------------------")
print(classes)

def create_bot_corpus(stem_words,classes):
    stem_words = sorted(list(set(stem_words)))
    classes= sorted(list(set(classes)))

    pickle.dump(stem_words,open('words.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))

    return stem_words,classes
stem_words,classes =create_bot_corpus(stem_words,classes)

print(stem_words)
print("-----------------------------")
print(classes)
print("-----------------------------")
print(word_tags_list[0])
print("-----------------------------")


training_data=[] # how are you?
number_of_tags=len(classes) # [4]
labels =[0]*number_of_tags # [0 1 0 0]

for word_tags in word_tags_list:
    bag_of_words=[]
    pattern_words=word_tags[0] # ["how are you?"] [0 1 2]
    for word in pattern_words:
        index= pattern_words.index(word)
        word= stemmer.stem(word.lower())
        pattern_words[index]=word
    for word in stem_words:
        if word in pattern_words:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    print(bag_of_words)


