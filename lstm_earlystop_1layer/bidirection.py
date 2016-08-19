import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator
import csv
import itertools
import numpy as np
import nltk
import time
import sys
import operator
import io
import array
from datetime import datetime

class GRUTheano:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim-3, word_dim))
        Ey_encode = np.zeros(3)
        Ey_decode = np.identity(3)
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim*2, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim*2, hidden_dim*2))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim*2))
        b = np.zeros((4, hidden_dim*2))
        c = np.zeros(3)
        U_decode = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (8, hidden_dim, hidden_dim))
        W_decode = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (8, hidden_dim, hidden_dim))
        b_decode = np.zeros((8, hidden_dim))
        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.Ey_encode = theano.shared(name='Ey_encode', value=Ey_encode.astype(theano.config.floatX))
        self.Ey_decode = theano.shared(name='Ey_decode', value=Ey_decode.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        self.U_decode = theano.shared(name='U_decode', value=U_decode.astype(theano.config.floatX))
        self.W_decode = theano.shared(name='W_decode', value=W_decode.astype(theano.config.floatX))
        self.b_decode = theano.shared(name='b_decode', value=b_decode.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mEy_encode = theano.shared(name='mEy_encode', value=np.zeros(Ey_encode.shape).astype(theano.config.floatX))
        self.mEy_decode = theano.shared(name='mEy_decode', value=np.zeros(Ey_decode.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        self.mU_decode = theano.shared(name='mU_decode', value=np.zeros(U_decode.shape).astype(theano.config.floatX))
        self.mW_decode = theano.shared(name='mW_decode', value=np.zeros(W_decode.shape).astype(theano.config.floatX))
        self.mb_decode = theano.shared(name='mb_decode', value=np.zeros(b_decode.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, Ey_encode, Ey_decode, V, U, W, b, c, U_decode, W_decode, b_decode = self.E, self.Ey_encode, self.Ey_decode, self.V, self.U, self.W, self.b, self.c, self.U_decode, self.W_decode, self.b_decode

        x = T.ivector('x')
        y = T.ivector('y')

        def forward_prop_step_encode_backward(x_t, s_t1_prev_b, c_t1_prev_b):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # Word embedding layer
            x_e_b = E[:,x_t]
            xy_e_b = theano.tensor.concatenate([x_e_b,Ey_encode], axis=0)

            #Encode   #LSTM Layer 1
            # i=z, f=r, add o,
            i_t1_b = T.nnet.hard_sigmoid(U[0].dot(xy_e_b) + W[0].dot(s_t1_prev_b) + b[0])
            f_t1_b = T.nnet.hard_sigmoid(U[1].dot(xy_e_b) + W[1].dot(s_t1_prev_b) + b[1])
            o_t1_b = T.nnet.hard_sigmoid(U[2].dot(xy_e_b) + W[2].dot(s_t1_prev_b) + b[2])
            g_t1_b = T.tanh(U[3].dot(xy_e_b) + W[3].dot(s_t1_prev_b) + b[3])
            c_t1_b = c_t1_prev_b*f_t1_b + g_t1_b*i_t1_b
            s_t1_b = T.tanh(c_t1_b)*o_t1_b

            return [s_t1_b, c_t1_b]

        def forward_prop_step_encode(x_t, s_t1_prev, c_t1_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # Word embedding layer
            x_e = E[:,x_t]
            xy_e = theano.tensor.concatenate([x_e,Ey_encode], axis=0)

            #Encode   #LSTM Layer 1
            # i=z, f=r, add o,
            i_t1 = T.nnet.hard_sigmoid(U[4].dot(xy_e) + W[4].dot(s_t1_prev) + b[4])
            f_t1 = T.nnet.hard_sigmoid(U[5].dot(xy_e) + W[5].dot(s_t1_prev) + b[5])
            o_t1 = T.nnet.hard_sigmoid(U[6].dot(xy_e) + W[6].dot(s_t1_prev) + b[6])
            g_t1 = T.tanh(U[7].dot(xy_e) + W[7].dot(s_t1_prev) + b[7])
            c_t1 = c_t1_prev*f_t1 + g_t1*i_t1
            s_t1 = T.tanh(c_t1)*o_t1
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            #o_t = T.nnet.softmax(V.dot(s_t3) + c)[0]

            return [s_t1, c_t1]

        def forward_prop_step_decode(x_t, y_t, s_t1_prev_d, c_t1_prev_d):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # Word embedding layer
            x_e = E[:, x_t]
            y_e = Ey_decode[:,y_t]
            xy_e_d = theano.tensor.concatenate([x_e, y_e], axis=0)

            # Decode   #LSTM Layer 1
            i_t1_d = T.nnet.hard_sigmoid(U_decode[0].dot(xy_e_d) + W_decode[0].dot(s_t1_prev_d) + b_decode[0])
            f_t1_d = T.nnet.hard_sigmoid(U_decode[1].dot(xy_e_d) + W_decode[1].dot(s_t1_prev_d) + b_decode[1])
            o_t1_d = T.nnet.hard_sigmoid(U_decode[2].dot(xy_e_d) + W_decode[2].dot(s_t1_prev_d) + b_decode[2])
            g_t1_d = T.tanh(U_decode[3].dot(xy_e_d) + W_decode[3].dot(s_t1_prev_d) + b_decode[3])
            c_t1_d = c_t1_prev_d * f_t1_d + g_t1_d * i_t1_d
            s_t1_d = T.tanh(c_t1_d) * o_t1_d

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o = T.nnet.softmax(V.dot(s_t1_d) + c)[0]

            return [o, s_t1_d, c_t1_d]

        def forward_prop_step_decode_test(x_t, o_t_pre_test, s_t1_prev_d_test, c_t1_prev_d_test):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # Word embedding layer
            x_e = E[:, x_t]
            #y_e = Ey[:, y_t]
            xy_e_d_test = theano.tensor.concatenate([x_e, o_t_pre_test], axis=0)

            # Decode   #LSTM Layer 1
            i_t1_d_test = T.nnet.hard_sigmoid(U_decode[0].dot(xy_e_d_test) + W_decode[0].dot(s_t1_prev_d_test) + b_decode[0])
            f_t1_d_test = T.nnet.hard_sigmoid(U_decode[1].dot(xy_e_d_test) + W_decode[1].dot(s_t1_prev_d_test) + b_decode[1])
            o_t1_d_test = T.nnet.hard_sigmoid(U_decode[2].dot(xy_e_d_test) + W_decode[2].dot(s_t1_prev_d_test) + b_decode[2])
            g_t1_d_test = T.tanh(U_decode[3].dot(xy_e_d_test) + W_decode[3].dot(s_t1_prev_d_test) + b_decode[3])
            c_t1_d_test = c_t1_prev_d_test * f_t1_d_test + g_t1_d_test * i_t1_d_test
            s_t1_d_test = T.tanh(c_t1_d_test) * o_t1_d_test


            o_test = T.nnet.softmax(V.dot(s_t1_d_test) + c)[0]

            return [o_test, s_t1_d_test, c_t1_d_test]

        [s_t1_b, c_t1_b], updates = theano.scan(
            forward_prop_step_encode_backward,
            sequences=x[::-1], #reverse y
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim))])

        [s_t1, c_t1], updates = theano.scan(
            forward_prop_step_encode,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim))])
        [o, s_t1_d, c_t1_d], updates = theano.scan(
            forward_prop_step_decode,
            sequences=[x,T.concatenate([[y[-1]],y[:-1]], axis=0)],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=theano.tensor.concatenate([s_t1[-1], s_t1_b[-1]], axis=0)),
                          dict(initial=theano.tensor.concatenate(c_t1[-1],c_t1_b[-1], axis=0))
                          ])

        [o_test, s_t1_d_test, c_t1_d_test], updates = theano.scan(
            forward_prop_step_decode_test,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros(3)),
                          dict(initial=theano.tensor.concatenate([s_t1[-1], s_t1_b[-1]], axis=0)),
                          dict(initial=theano.tensor.concatenate(c_t1[-1], c_t1_b[-1], axis=0))
                          ])
        prediction = T.argmax(o_test, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        # Total cost (could add regularization here)
        cost = o_error

        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)
        dU_decode = T.grad(cost, U_decode)
        dW_decode = T.grad(cost, W_decode)
        db_decode = T.grad(cost, b_decode)

        # Assign functions
        self.predict = theano.function([x], o_test)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc, dU_decode, dW_decode, db_decode])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2
        mU_decode = decay * self.mU_decode + (1 - decay) * dU_decode ** 2
        mW_decode = decay * self.mW_decode + (1 - decay) * dW_decode ** 2
        mb_decode = decay * self.mb_decode + (1 - decay) * db_decode ** 2

        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [],
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (U_decode, U_decode - learning_rate * dU_decode / T.sqrt(mU_decode + 1e-6)),
                     (W_decode, W_decode - learning_rate * dW_decode / T.sqrt(mW_decode + 1e-6)),
                     (b_decode, b_decode - learning_rate * db_decode / T.sqrt(mb_decode + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc),
                     (self.mU_decode, mU_decode),
                     (self.mW_decode, mW_decode),
                     (self.mb_decode, mb_decode)
                     ])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y) / float(num_words)

# utils
def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9,
                   callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
            num_examples_seen += 1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen)
                sys.stdout.write('.')
    return model

def save_model_parameters_theano(model, outfile):
    np.savez(outfile,
             E=model.E.get_value(),
             U=model.U.get_value(),
             W=model.W.get_value(),
             V=model.V.get_value(),
             b=model.b.get_value(),
             c=model.c.get_value(),
             U_decode=model.U_decode.get_value(),
             W_decode=model.W_decode.get_value(),
             b_decode=model.b_decode.get_value())
    print "Saved model parameters to %s." % outfile

def load_model_parameters_theano(path, modelClass=GRUTheano):
    npzfile = np.load(path)
    E, Ey_encode, Ey_decode, U, W, V, b, c = npzfile["E"], npzfile["Ey_encode"], npzfile["Ey_decode"], npzfile["U"], \
                                             npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"], npzfile[
                                                 "U_decode"], npzfile["W_decode"], npzfile["b_decode"]
    hidden_dim, word_dim = E.shape[0] + 3, E.shape[1]
    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    model.U_decode.set_value(U_decode)
    model.W_decode.set_value(W_decode)
    model.b_decode.set_value(b_decode)
    return model