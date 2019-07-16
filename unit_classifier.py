#import these moduls to use Myclassifier
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

'''
use like this 

predictor = Myclassifier(img_rows, img_cols) #make instance
predictor.clear()                            #initialte the model
predictor.fit( x_train, y_train)             #fitting the model
res = predictor.predict(x_test)              #predicting from x_test and return the list 

It did worked on my computer, with my own image data.
If you have any question, please contact to ilya-horiguchi@g.ecc.u-tokyo.ac.jp
See you on Monday Final class!

'''

batch_size = 10
epochs = 8
img_rows, img_cols = 1280, 640

class Myclassifier():
    def __init__(self,hight,width):
        self.results_train_ = {
            'loss': [],
            'accuracy': []
        }
        self.results_valid_ = {
            'loss': [],
            'accuracy': []
        }
        self.model = Sequential()
        self.hight = hight
        self.width = width
        self.optimizer=keras.optimizers.Adam()
        self.num_classes = 2 # 1 or -1 group
        
    def dup(self):
        return self
                  
    def clear(self):
        self.model = Sequential()
        self.model.add(contactv2D(32, kernel_size=(3, 3),
                activation='relu',
                data_format="channels_last",
                input_shape=(self.hight,self.width,3)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
        self.model.summary()
    def fit(self, x_train, y_train):
        #normalization
        self.model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
        self.model.fit(x=x_train,y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)
        
    def predict(self,x_test):
        return self.model.predict_classes(x_test)
        #y_results = [np.argmax(predictions[i,:]) for i in range(len(predictions))]
        #return y_results
        