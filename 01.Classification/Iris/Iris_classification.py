"""
120 / 4 / setosa / versicolor / virginica
Iris Column name
'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'

* gitgub에 올려두자!!!
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

"""Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR 해결 책."""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#gpu 메모리 할당 문제....
"""=============================="""

data_path = os.path.join('.', 'data')
train_path = os.path.join(data_path, 'iris_training.csv')
test_path = os.path.join(data_path, 'iris_test.csv')

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def load_data(y_name='Species'):
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_features, train_labels = train, train.pop(y_name)
    print(train_features)
    print(train_labels)
    train_features = train_features.values
    train_labels = train_labels.values

    train_features, validation_features, train_labels, validation_labels = train_test_split(train_features, train_labels, test_size=0.3)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES,header=0)
    test_features, test_labels = test, test.pop(y_name)
    print(test_features)
    print(test_labels)
    test_features = test_features.values
    test_labels = test_labels.values

    return (train_features, train_labels), (validation_features, validation_labels), (test_features, test_labels)

"""
load_data() 함수에서 return 부분에 ()를 안치고 할때는 이상하게 나오고 shape도 121,4  이런식으로 나왔는데
()를 하고 반환하니까 제대로 내가 원하는대로 나온다...신기하다.
"""


(train_features, train_labels), (validation_features, validation_labels), (test_features, test_labels) = load_data()

#train_features, validation_features, train_labels, validation_labels = train_test_split(train_features, train_labels, test_size=0.3)

print(train_features.shape)
print(train_labels.shape)
print(validation_features.shape)
print(validation_labels.shape)
print(test_features.shape)
print(test_labels.shape)

#data pipeline을 만들때 batch는 모두 다 동일애야함.
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).shuffle(buffer_size=10000).batch(10)
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels)).shuffle(buffer_size = 10000).batch(10)
test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(10) 

model = models.Sequential()
model.add(layers.Dense(256, input_dim=4, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

#hist = model.fit(train_features, train_labels, validation_data= (validation_features, validation_labels), epochs=100)
hist = model.fit(train_dataset, validation_data = validation_dataset, epochs=100)


model.save('iris.h5')

test_loss, test_acc = model.evaluate(test_dataset)

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='test loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='test acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper right')

plt.show()