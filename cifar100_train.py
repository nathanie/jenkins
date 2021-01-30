import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100

def get_model():
    inp = Input(shape=(32,32,3))
    x = Conv2D(32,3,activation='relu')(inp)
    x = Conv2D(32,3,activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(32,3,activation='relu')(x)
    x = Conv2D(32,3,activation='relu')(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
#     x = Dense(100,activation='relu')(x)
    x = Dense(10000,activation='relu')(x)
    x = Dense(100,activation='softmax')(x)
    m = Model(inp,x)
    m.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
    return m

(X_train, y_train), (X_val, y_val) = cifar100.load_data()
print((X_train.shape, y_train.shape), (X_val.shape, y_val.shape))

model = get_model()
model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=5)

from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
pred = model.predict(X_val)
print(classification_report(y_val,np.argmax(pred,axis=1)))