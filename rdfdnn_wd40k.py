import time
import numpy as np
import csv
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
import os

#keras import
from keras.models import Model, Sequential
from keras.layers import Dense, Input, concatenate, Embedding, merge, Activation, Reshape
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint


e_id = {}
with open('data/WD40k/entity2id.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        e_id[row[0]] = int(row[1])
        
r_id = {}
with open('data/WD40k/relation2id.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        r_id[row[0]] = int(row[1])
        
def load_triplet(path):
    ss = []
    os = []
    ps = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            s,o,p = row
            ss.append(e_id[s])
            os.append(e_id[o])
            ps.append(r_id[p])
    ss = np.array(ss)
    os = np.array(os)
    ps = np.array(ps)
    print(len(ss))
    return (ss,os,ps)
    
s_train, o_train, p_train = load_triplet("data/WD40k/train.txt")
s_test, o_test, p_test = load_triplet("data/WD40k/test.txt")
s_valid, o_valid, p_valid = load_triplet("data/WD40k/valid.txt")

#define model
inst_voc = len(e_id)
prop_voc = len(r_id)
entity_dim = 100
relation_dim = 100

print(inst_voc)
print(prop_voc)
print(len(s_train))

S_input = Input(shape=(1,), dtype="int32", name="S_inpput")
O_input = Input(shape=(1,), dtype="int32", name="O_inpput")

embed_layer = Embedding(inst_voc, entity_dim, W_regularizer=l2(0.001))
S_embed = embed_layer(S_input)
O_embed = embed_layer(O_input)

reshape_layer = Reshape((entity_dim,), input_shape=(1,entity_dim))
hidden_layer = Dense(entity_dim, activation="relu", W_regularizer=l2(0.001))
S_reshape = reshape_layer(S_embed)
O_reshape = reshape_layer(O_embed)
S_reshape = hidden_layer(S_reshape)
O_reshape = hidden_layer(O_reshape)

SO_merged = concatenate([S_reshape, O_reshape])
SO_merged = Dense(relation_dim, activation="relu", W_regularizer=l2(0.001))(SO_merged)

P_pred = Dense(prop_voc, activation="softmax")(SO_merged)

model = Model([S_input, O_input], P_pred)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
              
es_cb = EarlyStopping(monitor='sparse_categorical_crossentropy', min_delta=0.01, mode='auto')   
csv_cb = CSVLogger("train_log.txt", separator=',', append=True)
model.fit([s_train,o_train],p_train,epochs=100, callbacks=[es_cb,csv_cb])
model.save("wn40k_100.h5")

