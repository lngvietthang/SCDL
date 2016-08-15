#! /usr/bin/env python
from __future__ import division
import sys
import os
import time
import numpy as np
from utils import *
from datetime import datetime
from gru_theano import GRUTheano
from preprocess2 import load_data_from_json2, compute_f1, write_output, testing
import math

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "8000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "50"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "100"))
NEPOCH = int(os.environ.get("NEPOCH", "20"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", '../Data/Filippova/compression-data.json')
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "1000"))

ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
if not MODEL_OUTPUT_FILE:
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

# Load data
#x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)
(X_train, y_train, len_sent_train, sample_weight_train), (X_test, y_test, len_sent_test, sample_weight_test), (original_sentence_text, compression_sentence_text) = load_data_from_json2(INPUT_DATA_FILE,  0.2, VOCABULARY_SIZE)

'''
# Build model
print '\nBuild model'
model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)
#model = load_model_parameters_theano('GRU-2016-08-05-13-48-2000-50-100.dat.npz')

#Print SGD step time
print 'S2q'
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], LEARNING_RATE)
c = model.ce_error(X_train[10], y_train[10])
print ('loss x[10]: %f'%c)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()

# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen):
  dt = datetime.now().isoformat()
  loss = model.calculate_loss(X_train[:1000], y_train[:1000])
  print("\n%s (%d)" % (dt, num_examples_seen))
  print("--------------------------------------------------")
  print("Loss: %f" % loss)
  save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
  print("\n")
  sys.stdout.flush()

t3 = time.time()
for epoch in range(NEPOCH):
  print('Epoch %d: ' % epoch)
  train_with_sgd(model, X_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9,
    callback_every=PRINT_EVERY, callback=sgd_callback)
t4 = time.time()
print "SGD Train time: %f" % ((t4 - t3))
sys.stdout.flush()
#
print 'Testing...'
predict_test = testing(model, X_test)
np.save("%s.predict" % (MODEL_OUTPUT_FILE), predict_test)
'''

def compute_f1_except0(y_test, predict_test):
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
    f1[3] = f1[3] / (len(y_test) - num_error-num_f1_0)
    if (len(y_test)-num_error) ==0:
        print ('----- All predicts are zero -----')
    else:
        print ('Total: %d - Error sentences: %d = %d'%(len(y_test), num_error, len(y_test)-num_error))
        print ('No. sent F1 is 0: %d'%num_f1_0)
        print (f1[3])
        return f1

predict_test=np.load("/home/nhitt/Dropbox/Compression_lstm/rnn-tutorial-gru-lstm/Output_2ndAttention_theano/AttentionV2-GRU-2016-08-13-10-12-8000-50-100.dat.predict.npy")
print 'Compute f1:...'
f1 = compute_f1_except0(y_test, predict_test)
#write_output('./Output_seq2seq_theano', f1[5], original_sentence_text, compression_sentence_text, 0.6)

