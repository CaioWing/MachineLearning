import os
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from scipy import stats
import random


%%time
train = pd.read_parquet('../input/ubiquant-parquet/train_low_mem.parquet')
train.head()

investment_id = train.pop("investment_id")
investment_ids = list(investment_id.unique())
train = train.drop(['row_id', 'time_id'], axis = 1)
train['investment_id'] = investment_id

def create_dataset(train, investment_ids):
    index = np.zeros((len(investment_ids), 1))
    i = 0

    for element in investment_ids:
        index[i] = random.choice(train.index[train['investment_id'] == element].tolist())
        i = i+1

    index = np.reshape(index, (1, len(index)))
    df_sliced = train.take(index[0])
    
    y = df_sliced.loc[:, 'target']
    new_df = df_sliced.drop(['investment_id', 'target'], axis = 1)

    dataset = tf.data.Dataset.from_tensor_slices((new_df.values, y.values))
    train_dataset = dataset.shuffle(len(new_df)).batch(1)
    
    return train_dataset, y
    
def create_model():
    inputs = tf.keras.layers.Input((300, ), dtype=tf.float16)
    layers = tf.keras.layers.Dense(512, activation = 'sigmoid')(inputs)
    layers = tf.keras.layers.Dense(512, activation = 'sigmoid')(layers)
    layers = tf.keras.layers.Dense(256, activation = 'sigmoid')(layers)
    layers = tf.keras.layers.Flatten()(layers)
    layers = tf.keras.layers.Dense(128, activation = 'sigmoid')(layers)
    layers = tf.keras.layers.Dense(64, activation = 'sigmoid')(layers)
    output = tf.keras.layers.Dense(1)(layers)

    
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[inputs], outputs=[output])
    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy', rmse], optimizer = 'adam')

    
    return model
    
def fit_model(model, dataset, step):
    checkpoint = keras.callbacks.ModelCheckpoint(f"./model_{step}", save_best_only=False)
    early_stop = keras.callbacks.EarlyStopping(patience=10)
    history = model.fit(dataset, epochs = 1, batch_size = 32, callbacks=[checkpoint, early_stop])
    return history

def load_model(step):
    model = keras.models.load_model(f"./model_{step}")
    return model

def pearson(dataset, target):
    return stats.pearsonr(model.predict(dataset).ravel(), target.values)[0]
    
model = create_model()
model.summary()

model = create_model()

for step in range(0,300):
    dataset, y = create_dataset(train, investment_ids)
    if step > 1:
        load_model(step-1)
    fit_model(model, dataset, step)
    print('chute:',model.predict(dataset))
    print('valor correto', y)
    print('coef person is:', pearson(dataset, y))
    print('step number:', step)
