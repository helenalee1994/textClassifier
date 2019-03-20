# author - Richard Liao + Helena Lee
# Dec 26 2016 / Mar 07 2019
from __future__ import division

import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, optimizers
from nltk import tokenize
#### additional
import pickle
from args import get_parser
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
from tensorboardX import SummaryWriter
from sys import exit
writer = SummaryWriter()
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# # =============================================================================

def main():
    ### select GPU
    gpu_id = opts.gpu
    print('Current running on GPU number:', gpu_id)
    gpu_options = tf.GPUOptions(visible_device_list=str(gpu_id))
    config = tf.ConfigProto(device_count = {'GPU': gpu_id, 'CPU': 10},
                            gpu_options = gpu_options,
                            intra_op_parallelism_threads = 32,
                            inter_op_parallelism_threads = 32)
    sess = tf.Session(config = config)
    K.set_session(sess)
    ####
    MAX_SENT_LENGTH = 100
    MAX_SENTS = 15
    MAX_NB_WORDS = 20000

    ##### customized settings ##### 
    p = os.path.abspath(opts.snapshots)
    tag = '/'.join(p.split(os.sep)[3:])# store the path, but w/o prefix workspace/dir_HugeFiles
    print('tag of tensor board: %s'%(tag))
    
    reviews, labels = load_pickle(opts.train)
    
    # if want to use less data to train
    small = opts.small
    if small:
        reviews, labels = reviews[:5000], labels[:5000]
    texts = [' '.join(recipe) for recipe in reviews]
    '''
    # sentences = list of string, each string contains one sentence
    # texts =  flatten sentences, separate by recipes
    # reviews = list of sentences
    reviews = [v['directions'] for v in dic.values()]
    texts = [' '.join(v['directions']) for v in dic.values()]
    labels = [v['GI'] for v in dic.values()]    
    '''

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1

    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify = labels, test_size = 0.2, random_state = opts.random)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.25, random_state = 1 + opts.random)
    
    print('Number of positive and negative reviews in traing and validation set')
    print y_train.sum(axis=0)
    print y_val.sum(axis=0)
    
    # delete variable to release memory
    del data
    
    class_wights = opts.pweight
    # if -1, then automatically caculate the balanced weight
    if class_wights == -1:
        class_01 = y_train.sum(axis= 0)
        class_weights = round(class_01[0]/class_01[1],1)
    print('class weight is %.1f' % class_weights)
    
    if opts.foodW2V:
        pretrained = Word2Vec.load(opts.foodW2V)# '../data/foodvec300.model'
        embeddings_index = pretrained.wv
        EMBEDDING_DIM = pretrained.vector_size
        opts.emdding_dim = EMBEDDING_DIM
        print(opts.foodW2V)
    
    '''
    embeddings_index = {}
    f = open(opts.gloveW2V)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))  
    '''
    
    # building Hierachical Attention network
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in pretrained.wv.vocab:
            embedding_vector = embeddings_index.get_vector(word)
        else:
            # words not found in embedding index will be UNK
            embedding_vector = embeddings_index.get_vector('UNK')
        embedding_matrix[i] = embedding_vector
    
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SENT_LENGTH,
                                trainable=True,
                                mask_zero=True)

    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(100)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
    l_att_sent = AttLayer(100)(l_lstm_sent)
    preds = Dense(2, activation='softmax')(l_att_sent)
    model = Model(review_input, preds)
    
    #SGD
    custom_optimizer = optimizers.SGD(lr=opts.lr, momentum= 0.9)
    if opts.optm:
        custom_optimizer = opts.optm
    model.compile(loss='binary_crossentropy', 
                  optimizer = custom_optimizer, #'rmsprop',
                  metrics=['acc'])
    
    if opts.resume:
        modelname = opts.resume+'_model.h5'
        statename = opts.resume+'.pickle'
        if os.path.isfile(modelname):
            print("=> loading checkpoint '{}'".format(opts.resume))
            model.load_weights(modelname)
            state = load_pickle(statename)
            best_val = state['best_val']
        else:
            raise NameError("=> no checkpoint found at '{}'".format(modelname))
    else:
        best_val = float('inf')
        state = dict()
        
    if opts.test:
        modelname = opts.test+'_model.h5'
        statename = opts.test+'.pickle'
        if os.path.isfile(modelname):
            print("=> loading checkpoint '{}'".format(opts.test))
            model.load_weights(modelname)
            state = load_pickle(statename)
            best_val = state['best_val']
        else:
            raise NameError("=> no checkpoint found at '{}'".format(modelname))
        model.threshold = state['threshold']
        print('train')
        test_f1 = test(X_train, y_train, model, threshold = model.threshold, print_ = True)
        print('val')
        test_f1 = test(X_val, y_val, model, threshold = model.threshold, print_ = True)
        print('test')
        test_f1 = test(X_test, y_test, model, threshold = model.threshold, print_ = True)
        exit()

    print("model fitting - Hierachical attention network")
    valtrack = 0
    
    if hasattr(state, 'epoch'):
        start_epoch = state['epoch']
    else:
        start_epoch = 1
    
    for current_epoch in range(start_epoch, opts.epochs):
        state['epoch'] = current_epoch
        print('epoch', state['epoch'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=1, batch_size=opts.batch_size, class_weight={0:1.0,1:class_weights})
        # slowest version
        num_grid = 20
        thresholds = [i/num_grid for i in range(1, num_grid, 1)]

        # improved version
        if hasattr(model, 'threshold'):
            t = model.threshold
        else:
            t = 0.5

        # need to try model probability threshold 
        thresholds = [t-1/num_grid, t, t+1/num_grid]

        # get the probability of being poisitve
        prob_class1 = model.predict(X_val)[:,1]
        dic_val = {t: validate(prob_class1, y_val, threshold = t, print_ = True) for t in thresholds}

        # find the loweset negative F1
        bests = {k: abs(k-0.5) for k,v in dic_val.iteritems() if v == min(dic_val.values())}

        # choose the closest to 0.5 if receive equal F1 
        best_threshold = [k for k,v in bests.iteritems() if v == min(bests.values())][0]
        model.threshold = best_threshold
        val_loss = dic_val[best_threshold]
        val_f1 = - val_loss
        writer.add_scalar(tag+'/f1_validation', val_f1, state['epoch'])

        print('test')
        test_f1 = test(X_test, y_test, model, threshold = model.threshold, print_ = True)
        
        # is best or not
        is_best = val_loss < best_val
        best_val = min(val_loss, best_val)
        state['best_val'] = best_val
        if is_best:
            valtrack = 0
            filepath = opts.snapshots
            filepath += 'model_e%03d_v-%.3f' % (state['epoch'], state['best_val'])
            make_dir(filepath)
            model.save(filepath+'_model.h5')
            state['opts']=opts
            state['threshold']=model.threshold
            save_pickle(filepath+'.pickle',state)
        else:
            valtrack+=1
        print '** Validation: %f (best) - %d (valtrack)' % (best_val, valtrack)
        
def save_pickle(filename, obj, overwrite = False):
    make_dir(filename)
    if os.path.isfile(filename) == True and overwrite == False:
        print('already exists'+filename)
    else:
        with open(filename, 'wb') as gfp:
            pickle.dump(obj, gfp, protocol=2)
            gfp.close()
            
def make_dir(filename):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('make dir')
        
def load_pickle(filename):
    with open(filename, 'rb') as gfp:
        r = pickle.load(gfp)
    return r

# ref: http://keras.io/layers/writing-your-own-keras-layers/
class AttLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def test(X_test, y_test, model, threshold, print_ = True):
    prob_class1 = model.predict(X_test)[:,1]
    return validate(prob_class1, y_test, threshold, print_ = True)

def validate(prob_class1, y_test, threshold, print_ = True):
    preds = [1 if i>threshold else 0 for i in prob_class1]
    true = y_test.argmax(axis = -1).tolist()
    f1 = f1_score(true, preds)
    if print_ == True:
        print(':::current prob threshold %.3f '%(threshold))
        print('   positive number: pred %d, true %d' %(sum(preds), sum(true)))
        print('   -f1 %.3f, precision %.3f, recall %.3f' % (f1, precision_score(true, preds), recall_score(true, preds)))
    return -f1 # return negative

if __name__ == '__main__':
	main()