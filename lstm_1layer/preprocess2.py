from __future__ import division
import sys
import pickle
import io
import json
import os
from itertools import izip
import nltk.data
import numpy as np
#from keras.preprocessing import sequence
from numpy import array
import itertools
import math

SENTENCE_START_TOKEN = 'eos'
UNKNOWN_TOKEN = 'unk'

def word2vec(sent, word2index_dict):
    '''
    Word2vec of a sentence
    :param sent: input sentence
    :param word2vec_dict: dict of word2vec
    :param maxlen: max len of sentence in dataset
    :return: vector of sentence (list vector of words)
    '''
    sent = "%s %s" % (SENTENCE_START_TOKEN, sent)
    words_in_sent = [x for x in nltk.word_tokenize(sent)]
    i = len(words_in_sent)
    array_sent=[0]*i
    sample_weight = [0]*i
    for j in range(i):
        if words_in_sent[j].lower() not in word2index_dict.keys():
            words_in_sent[j] = UNKNOWN_TOKEN
        array_sent[j] = (word2index_dict[words_in_sent[j].lower()])
        sample_weight[j] = 1
    return ((array_sent),array(sample_weight))

def label_compress(sent, comp):
    '''
    Label compressed of sentence
    :param sent: original sentence
    :param comp: compressed sentence
    :return: list of label (0 or 1) of each word of sentence
    '''
    sent = "%s %s" % (SENTENCE_START_TOKEN, sent)
    words_in_sent = [x for x in nltk.word_tokenize(sent)]
    i = len(words_in_sent)
    l= [0]*i
    for j in range(i):
        if words_in_sent[j].lower() in [comp_word.lower() for comp_word in nltk.word_tokenize(comp)]:
            l[j] = 1
    l[0]=2
    return (l,i)

# read json input
def read_json_file(path_to_json):
    objects = []
    data = ''
    with io.open(path_to_json, 'r', encoding='utf8') as f:
        for line in f:
            if line in ['\n', '\n\r']:
                objects.append(json.loads(data))
                data = ''
            else:
                data += line
        try:
            objects.append(json.loads(data))
        except:
            return objects
    return objects

# get original sentence, compression sentence
def get_originalSent_compressionSent(object):
    return (object['graph']['sentence'], object['compression']['text'])

def word2index(json_objects, vocabulary_size):
    sentences=[]
    unknown_token = UNKNOWN_TOKEN
    sentence_start_token = SENTENCE_START_TOKEN
    for object in json_objects:
        original_sentence, _ = get_originalSent_compressionSent(object)
        sentences.append(original_sentence)
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    #print "Found %d unique words tokens." % len(word_freq.items())
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 2)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    index_to_word.append(sentence_start_token)
    #print(len(index_to_word))
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    return (word_to_index, index_to_word)

def load_data_from_json2(path_to_json, test_split, vocabulary_size):
    '''
    Load data for training and testing from json file
    :param path_to_json: path to json file
    :param word2vec_dict: dictionary of word2vec
    :return: X_train, y_train, X_test, y_test
    '''
    X=[]
    y=[]
    len_sent_array=[]
    sample_weight=[]
    objests=read_json_file(path_to_json)
    print 'Data %d sentences'%len(objests)
    i=0
    original_sentence_array=[]
    compression_sentence_array=[]
    word2indext_dict, _ = word2index(objests, vocabulary_size)
    for object in objests:
        original_sentence, compression_sentence = get_originalSent_compressionSent(object)
        (array_sent, sample_w) = word2vec(original_sentence, word2indext_dict)
        X.append(array_sent)
        sample_weight.append(sample_w)
        (y_l,l) = label_compress(original_sentence, compression_sentence)
        y.append(y_l)
        len_sent_array.append(l)
        i+=1
        if i%100==0:
            sys.stdout.write('.')
        #get text array:
        original_sentence_array.append(original_sentence)
        compression_sentence_array.append(compression_sentence)
    return ((X[int(len(X)*test_split):],y[int(len(y)*test_split):], len_sent_array[int(len(len_sent_array)*test_split):], sample_weight[int(len(sample_weight)*test_split):]), (X[:int(len(X)*test_split)], y[:int(len(y)*test_split)], len_sent_array[:int(len(len_sent_array)*test_split)], sample_weight[:int(len(sample_weight)*test_split)]), (original_sentence_array, compression_sentence_array))

def testing(model, X_test):
    predict_y_test=[]
    for i in range(len(X_test)):
        predict_test=model.predict_class(X_test[i])
        predict_y_test.append(predict_test)
    return predict_y_test

def compute_f1(y_test, predict_test):
    f1=[0,0,0,0,0,[]]
    num_error=0
    num_f1_0 = 0
    for i in range(len(y_test)):
        num_True = 0
        num_predict = 0
        num_test = 0
        for y in range(1,len(y_test[i])):
            if y_test[i][y] == predict_test[i][y] and y_test[i][y] == 1:
                num_True+=1
            if y_test[i][y] ==1:
                num_test+=1
            if predict_test[i][y] ==1:
                num_predict+=1
        f1[0]+=num_True
        f1[1]+=num_predict
        f1[2]+=num_test
        if num_True ==0:
            f1_sent = 0
            num_f1_0+=1
        else:
            precision = num_True/num_predict
            recall = num_True/num_test
            f1_sent= 2*(precision*recall)/(precision+recall)
        if math.isnan(f1_sent):
            f1_sent=0
            num_error+=1
        f1[3]+=f1_sent

        f1[4]+=len(y_test[i])
        f1[5].append((i, f1_sent, predict_test[i], y_test[i]))
    f1[3] = f1[3] / (len(y_test) - num_error)
    if (len(y_test)-num_error) ==0:
        print ('----- All predicts are zero -----')
    else:
        print ('Total: %d - Error sentences: %d = %d'%(len(y_test), num_error, len(y_test)-num_error))
        print ('No. sent F1 is 0: %d'%num_f1_0)
        print (f1[3])
        return f1

def write_output(output_folder, output_index, text_ori, text_compress, threshold):
    #train_index=8000
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with io.open(os.path.join(output_folder, 'text_compression>'+str(threshold)), 'w' , encoding='utf8') as flarger, io.open(os.path.join(output_folder, 'text_compression<'+str(threshold)), 'w', encoding='utf8') as fsmaller, io.open(os.path.join(output_folder, 'text_compression_all'), 'w', encoding='utf8') as fall, io.open(os.path.join(output_folder, 'text_compression<<'+str(threshold)), 'w', encoding='utf8') as fsmallest:
        for (i, f1_sent, predict_test, y_test) in output_index:
            out='original: %s\nsystem: %s\ngold: %s\ngold :%s\n\n' % (text_ori[i], list(predict_test), list(y_test), text_compress[i])
            fall.write(out)
            if f1_sent>=threshold:
                flarger.write(out)
            else:
                fsmaller.write(out)
            if f1_sent<(threshold/2):
                fsmallest.write(out)

if __name__ == '__main__':
    (X_train, y_train, len_sent_train, sample_weight_train), (X_test, y_test, len_sent_test, sample_weight_test), (
    original_sentence_text, compression_sentence_text) = load_data_from_json2('../Data/SentenceCompression/Filippova/compression-data-100k.json',  0.3, 8000)
    # json_objects= read_json_file('../../Data/SentenceCompression/Filippova/compression-data.json')
    # print statistic_len(json_objects)
    print('y_train len: ', len(y_train) , len(y_test))
    print('X_train:', X_train[0])
    print('y_test: ', y_test[0])
    print('X_test:', X_test[0])
