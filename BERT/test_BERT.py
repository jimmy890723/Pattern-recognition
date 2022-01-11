#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
train = pd.read_csv("./nlp_final/train.csv")
test = pd.read_csv("./nlp_final/test.csv")
train.head()


# In[5]:

import wget

site_url = 'https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py'
wget.download(site_url)

#get_ipython().system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')
#!pip install sentencepiece


# In[7]:


import tensorflow_hub as hub
import tokenization
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[8]:


X = np.array(train['text'])
y = np.array(train['target'])


# In[9]:


def bert_encode(texts,tokenizer, max_len = 512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        pad_len = max_len - len(input_sequence)

        tokens += [0]*pad_len
        pad_mask = [1]*len(input_sequence)+[0]*pad_len

        segment_id = [0]*max_len

        all_tokens.append(tokens)
        all_masks.append(pad_mask)
        all_segments.append(segment_id)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# In[10]:


def build_model(bert_layer,max_len=512):

    input_word_ids = tf.keras.layers.Input(shape=(max_len,),dtype=tf.int32,name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_len,),dtype=tf.int32,name="input_mask")
    input_segment_ids = tf.keras.layers.Input(shape=(max_len,),dtype=tf.int32,name="input_segment_ids")

    _,sequence_output = bert_layer([input_word_ids,input_mask,input_segment_ids])
    clf_output = sequence_output[:,0,:]
    model_X = tf.keras.layers.Dense(100,activation='relu')(clf_output)
    model_X = tf.keras.layers.BatchNormalization()(model_X)
    model_X = tf.keras.layers.Dropout(0.5)(model_X)
    model_X = tf.keras.layers.Dense(100,activation='relu')(model_X)
    model_X = tf.keras.layers.BatchNormalization()(model_X)
    model_X = tf.keras.layers.Dropout(0.5)(model_X)
    model_X = tf.keras.layers.Dense(100,activation='relu')(model_X)
    model_X = tf.keras.layers.BatchNormalization()(model_X)
    model_X = tf.keras.layers.Dropout(0.5)(model_X)
    out = tf.keras.layers.Dense(1,activation='sigmoid')(model_X)

    model = tf.keras.models.Model(inputs=[input_word_ids,input_mask,input_segment_ids],outputs=out)

    return model


# In[13]:

import time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
for i in range(10):
    try:
        bert_layer = hub.KerasLayer(module_url, trainable=True)
    except Exception as e:
        if i >= 9:
            time.sleep(1.0)
        else:
            time.sleep(0.5)
    else:
        time.sleep(0.1)
        break



# In[14]:


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
train_input = bert_encode(X_train,tokenizer,max_len=132)
val_input = bert_encode(X_test,tokenizer,max_len=132)
test_input = bert_encode(test.text.values,tokenizer,max_len=132)
train_labels = y_train
val_labels = y_test


# In[16]:


model = build_model(bert_layer,max_len=132)
model.summary()


# In[17]:


val_input[0].shape


# In[18]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam


# In[ ]:


checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
es = EarlyStopping(monitor='val_loss',patience=3,verbose=1,restore_best_weights=True,min_delta=0.01)

model.compile(optimizer=Adam(lr=1e-5),loss='binary_crossentropy',metrics=['accuracy'])
train_history = model.fit(
    train_input, train_labels,
    validation_data=(val_input,val_labels),
    epochs=30,
    callbacks=[checkpoint,es],
    batch_size=8
)


# In[ ]:


import matplotlib.pyplot as plt

fig = plt.figure(1)
plt.plot(train_history.history['loss'], label='train')
plt.plot(train_history.history['val_loss'], label='test')
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss_rate')
plt.legend()
#plt.show()
fig.savefig('Loss.png')

# In[ ]:


fig2 = plt.figure(2)
plt.plot(train_history.history['accuracy'], label='train')
plt.plot(train_history.history['val_accuracy'], label='test')
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy_rate')
plt.legend()
#plt.show()
fig2.savefig('Accuracy.png')


# In[ ]:


y_pred = model.predict(test_input)

ans = pd.DataFrame({'id':np.array(test['id']),'target':np.array(y_pred.round().astype(int)).reshape(-1)})
ans.to_csv('submission.csv',index=False)

