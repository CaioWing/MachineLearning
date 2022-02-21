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

def create_dataset(train, investment_ids):
    index = np.zeros((len(investment_ids), 1))
    i = 0

    for element in investment_ids:
        index[i] = random.choice(train.index[train['investment_id'] == element].tolist())
        i = i+1

    index = np.reshape(index, (1, len(index)))
    df_sliced = train.take(index[0])
    
    y = df_sliced.loc[:, 'target']
    investment_id_input = df_sliced['investment_id']
    
    #investment_id_input = tf.cast(investment_id_input, dtype = 'float32', name=None)
    
    investment_id_input = pd.DataFrame(np.array([investment_id_input]))
    
    new_df = df_sliced.drop(['investment_id', 'target'], axis = 1)

    dataset = tf.data.Dataset.from_tensor_slices(((new_df.values, investment_ids), y.values))
    train_dataset = dataset.shuffle(len(new_df)).batch(1)
    
    #size_investment = np.size(investment_ids)
    #investment_id_lookup_layer = tf.keras.layers.IntegerLookup(max_tokens=size_investment)
    #investment_id_lookup_layer.adapt(investment_ids)
    
    return dataset, y, investment_id_input
    
def create_model():
    
    #investment_id_lookup_layer.adapt(investment_ids)
    
    investment_id_inputs = tf.keras.Input((1, ), dtype=tf.float32)
    inputs = tf.keras.layers.Input((300, ), dtype=tf.float32)
    
    layersi = investment_id_lookup_layer(investment_id_inputs)
    layersi = tf.keras.layers.Embedding(size_investment, 32, input_length=1)(layersi)
    layersi = tf.keras.layers.Reshape((-1, ))(layersi)
    layersi = tf.keras.layers.Dense(64, activation='sigmoid')(layersi)
    layersi = tf.keras.layers.Dense(64, activation='sigmoid')(layersi)
    layersi = tf.keras.layers.Dense(64, activation='sigmoid')(layersi)
    
    layers = tf.keras.layers.Dense(512, activation = 'sigmoid')(inputs)
    layers = tf.keras.layers.Dense(512, activation = 'sigmoid')(layers)
    layers = tf.keras.layers.Dense(256, activation = 'sigmoid')(layers)
    layers = tf.keras.layers.Flatten()(layers)
    layers = tf.keras.layers.Dense(128, activation = 'sigmoid')(layers)
    layers = tf.keras.layers.Dense(64, activation = 'sigmoid')(layers)

    
    final_layer = tf.keras.layers.Concatenate(axis=1)([layersi, layers])
    final_layer = tf.keras.layers.Dense(512, activation='sigmoid')(final_layer)
    final_layer = tf.keras.layers.Dense(128, activation='sigmoid')(final_layer)
    final_layer = tf.keras.layers.Dense(32, activation='sigmoid')(final_layer)
    output = tf.keras.layers.Dense(1)(final_layer)

    
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[inputs, investment_id_inputs], outputs=[output])
    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy', rmse, 'mape'], optimizer = 'adam')

    return model
    
def fit_model(model, dataset, step):
    checkpoint = keras.callbacks.ModelCheckpoint(f"./model_{step}", save_best_only=False, save_weights_only=True)
    early_stop = keras.callbacks.EarlyStopping(patience=10)
    history = model.fit(dataset, epochs = 1, batch_size = 32, callbacks=[checkpoint, early_stop])
    return history

def load_model(step):
    model = keras.models.load_model(f"./model_{step}")
    return model

def pearson(dataset, target):
    return stats.pearsonr(model.predict(dataset).ravel(), target.values)[0]

%%time
train = pd.read_parquet('../input/ubiquant-parquet/train_low_mem.parquet')
train.head()

investment_id = train.pop("investment_id")
investment_ids = list(investment_id.unique())
train = train.drop(['row_id', 'time_id'], axis = 1)
train['investment_id'] = investment_id
investment_ids = np.array(investment_ids)

size_investment = np.size(investment_ids)
investment_id_lookup_layer = layers.IntegerLookup(max_tokens=size_investment)
investment_id_lookup_layer.adapt(investment_ids)

model = create_model()
model.summary()

for step in range(0,300):
    train_dataset, y, investment_id_input = create_dataset(train, investment_ids)
    if step > 0:
        load_model(step-1)
    fit_model(model, train_dataset, step)
    #print('chute:',model.predict(dataset))
    #print('valor correto', y)
    print('coef person is:', pearson(dataset, y))
    print('step number:', step)
    del investment_id_inputs
    del y
    del train_dataset
