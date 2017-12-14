from knowledge.wnmanager import WNManager
wn = WNManager("data/wordnet/wordnet-mlj12-definitions.txt", "data/wordnet/wordnet-mlj12-train.txt", "data/wordnet/wordnet-mlj12-test.txt")
s_train, o_train, p_train = wn.get_train_data()
s_test, o_test, p_test = wn.get_test_data()

import os
import sys
import time
import argparse
import pdb
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, accuracy_score

#keras import
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Merge, Embedding, merge, Activation, Reshape
from keras import backend as K
from keras.regularizers import l2

#define model
inst_voc = wn.n_inst
prop_voc = wn.n_prop
entity_dim = 30
relation_dim = 30

S_input = Input(shape=(1,), dtype="int32", name="S_inpput")
O_input = Input(shape=(1,), dtype="int32", name="O_inpput")

embed_layer = Embedding(inst_voc, entity_dim, W_regularizer=l2(0.001))
S_embed = embed_layer(S_input)
O_embed = embed_layer(O_input)

reshape_layer = Reshape((entity_dim,), input_shape=(1,entity_dim))
hidden_layer = Dense(entity_dim, activation="tanh", W_regularizer=l2(0.001))
S_reshape = reshape_layer(S_embed)
O_reshape = reshape_layer(O_embed)
S_reshape = hidden_layer(S_reshape)
O_reshape = hidden_layer(O_reshape)

SO_merged = merge([S_reshape, O_reshape], mode="concat")
SO_merged = Dense(relation_dim, activation="tanh", W_regularizer=l2(0.001))(SO_merged)

P_pred = Dense(prop_voc, activation="softmax")(SO_merged)

model = Model([S_input, O_input], P_pred)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#train
model.fit([s_train,o_train],p_train,epochs=10)

#save
weights = model.get_weights()
pd.to_pickle(weights, "wn18_weights.pkl")

#test
p_pred = model.predict([s_test, o_test])
acc = []
for k in range(1,10):
    acc.append(wn.topk_accuracy(None, p_pred, p_test, k))
