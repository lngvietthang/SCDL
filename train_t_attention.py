#! /usr/bin/env python

import sys
import os
import time
import numpy as np
from utils_t_attention import *
from datetime import datetime
from t_attention_theano import GRUTheano
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
  MODEL_OUTPUT_FILE = "tAttention-GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

# Load data
#x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)
(X_train, y_train, len_sent_train, sample_weight_train), (X_test, y_test, len_sent_test, sample_weight_test), (original_sentence_text, compression_sentence_text) = load_data_from_json2(INPUT_DATA_FILE,  0.2, VOCABULARY_SIZE)

# Build model
print '\nBuild model'
model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)
#model = load_model_parameters_theano('GRU-2016-08-05-13-48-2000-50-100.dat.npz')

#Print SGD step time
print 'tAttention'
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
print "tAttention SGD Train time: %f" % ((t4 - t3))
sys.stdout.flush()
#
print 'Testing...'
predict_test = testing(model, X_test)
np.save("%s.predict" % (MODEL_OUTPUT_FILE), predict_test)

print 'Compute f1:...'
f1 = compute_f1(y_test, predict_test)
write_output('./Output_tAttention_theano', f1[5], original_sentence_text, compression_sentence_text, 0.6)

