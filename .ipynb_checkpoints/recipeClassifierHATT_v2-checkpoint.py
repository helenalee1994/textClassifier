# author - Richard Liao
# Dec 26 2016
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
from keras import initializers
from nltk import tokenize

#### load my modules
import pickle
def load_pickle(filename):
    with open(filename, 'rb') as gfp:
        r = pickle.load(gfp)
    return r
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
### useful for class weight
import functools
from itertools import product
import tensorflow as tf

###
from tensorboardX import SummaryWriter
writer = SummaryWriter()

### select GPU
gpu_id = 5
print('Current running on GPU number:', gpu_id)
#gpu_options = tf.GPUOptions(visible_device_list=str(gpu_id))
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

config = tf.ConfigProto( device_count = {'GPU': gpu_id, 'CPU': 10} )
sess = tf.Session(config = config)
K.set_session(sess)
####

MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
tag = 'recipe_v2'

### load my data
dir_HugeFiles = '../../dir_HugeFiles/'
dir_save = os.path.normpath(dir_HugeFiles+'processed_data_0306/GId.pickle')
reviews, labels = load_pickle(dir_save)
small = False
if small:
    reviews, labels = reviews[:500], labels[:500]
texts = [' '.join(recipe) for recipe in reviews]
'''
### overwrite review, text, labels
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
'''
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
'''

X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify = labels, test_size = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.25)

print('Number of positive and negative reviews in traing and validation set')
print y_train.sum(axis=0)
print y_val.sum(axis=0)

GLOVE_DIR = "../../dir_HugeFiles/glove.6B"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

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

def batch_f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# apply weights
# ref: https://github.com/keras-team/keras/issues/2115
class_weights = 58.7
def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):

        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask
w_array = np.ones((2,2))
w_array[0,0] = class_weights
w_array[1,1] = class_weights
ncce = functools.partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'

model.compile(loss = ncce, #loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical attention network")

epochs = 10000

def evaluation(true, prob):
    #preds = prob.argmax(axis = -1).tolist()
    #true = true.argmax(axis = -1).tolist()
    '''
    true: list of int (0 or 1)
    '''
    f1 = f1_score(true, preds)
    print('   positive number: pred %d, true %d' %(sum(preds), sum(true)))
    print('   -f1 %.3f, precision %.3f, recall %.3f' % (f1, precision_score(true,preds), recall_score(true, preds)))
    
def test(X_test, y_test, model, threshold, print_ = True):
    prob_class0 = model.predict(X_test)[:,0]
    preds = [1 if i>threshold else 0 for i in prob_class0]
    true = y_test.argmax(axis = -1).tolist()
    f1 = f1_score(true, preds)
    if print_ == True:
        print(':::current prob threshold %.3f '%(threshold))
        print('   positive number: pred %d, true %d' %(sum(preds), sum(true)))
        print('   -f1 %.3f, precision %.3f, recall %.3f' % (f1, precision_score(true, preds), recall_score(true, preds)))
    return -f1 # return negative

def validate(prob_class0, y_test, threshold, print_ = True):
    preds = [1 if i>threshold else 0 for i in prob_class0]
    true = y_test.argmax(axis = -1).tolist()
    f1 = f1_score(true, preds)
    if print_ == True:
        print(':::current prob threshold %.3f '%(threshold))
        print('   positive number: pred %d, true %d' %(sum(preds), sum(true)))
        print('   -f1 %.3f, precision %.3f, recall %.3f' % (f1, precision_score(true, preds), recall_score(true, preds)))
    return -f1 # return negative

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

best_val = 0
valtrack = 0

for current_epoch in range(epochs):
    state ={'epoch': current_epoch+1}
    print('epoch', state['epoch'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              nb_epoch=1, batch_size=50)
    # slowest version
    num_grid = 20
    thresholds = [i/num_grid for i in range(1, num_grid, 1)]
    
    # improved version
    if hasattr(model, 'threshold'):
        t = model.threshold
    else:
        t = 0.5
    
    # need to try model probability threshold 
    if valtrack%10 == 0 or state['epoch']>3:
        thresholds = [t-1/num_grid, t, t+1/num_grid]
    prob_class0 = model.predict(X_val)[:,0]
    dic_val = {t: validate(prob_class0, y_val, threshold = t, print_ = True) for t in thresholds}
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
    
    is_best = val_loss < best_val
    best_val = min(val_loss, best_val)
    state['best_val'] = best_val
    valtrack+=1
    print '** Validation: %f (best) - %d (valtrack)' % (best_val, valtrack)

    if is_best:
        valtrack = 0
        filepath = 'save_models/'
        filepath += 'model_e%03d_v-%.3f' % (state['epoch'], state['best_val']) 
        model.save(filepath+'_model.h5')
        save_pickle(filepath+'.pickle',state)
