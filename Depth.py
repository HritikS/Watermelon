#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Data is boss - Sheet1.csv')


# In[3]:


df.head()


# In[4]:


train_dataset = df.sample(frac=0.8,random_state=0)
test_dataset = df.drop(train_dataset.index)


# In[5]:


train_stats = train_dataset.describe()
train_stats.pop('Depth (0-30mm) Output')
train_stats = train_stats.transpose()
train_stats


# In[6]:


train_labels = train_dataset.pop('Depth (0-30mm) Output')
test_labels = test_dataset.pop('Depth (0-30mm) Output')


# In[7]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# In[8]:


def build_model():
    model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
  ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model


# In[9]:


model = build_model()


# In[10]:


model.summary()


# In[11]:


EPOCHS = 100
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=1, callbacks=[early_stop])


# In[12]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.head(100)


# In[13]:


test_predictions = model.predict(normed_test_data).flatten()


# In[14]:


test_predictions


# In[15]:


test_labels


# In[16]:


model.save('weights.h5')


# In[ ]:
def get():
    x, y = train_stats['mean'], train_stats['std']
    return (x, y)


