# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:02:56 2019

@author: Rahul
"""

# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wj-aNzWsigjcnczEjz1Qae2X1AtbjWW6
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
#from keras.datasets import mnist
from keras import *
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import glob
import time
from skimage import io
import os
from imageio import imread
from skimage.transform import resize
from keras import regularizers
import csv
from keras.applications.inception_v3 import InceptionV3, preprocess_input

datasets_path = './101_ObjectCategories/101_ObjectCategories' #Add the path to the unzipped folder

def load_images(path,n=0):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    i=-1
    labels = []
    for label in os.listdir(path):
        imageCount = 0
        back_path = path + '/' + label
        labels.append(label)
        i = i+1
        count = 0
        for filename in os.listdir(back_path):
            imageCount += 1
        for filename in os.listdir(back_path):
            image_path = path + '/' + label + '/' + filename
            print(image_path)
#            os.path.join(back_path,filename)
            img = image.load_img(image_path,target_size=(224,224))
            if count < 30:
                img = image.img_to_array(img)
                #Y.append(image)
                #image = imresize(image,[128,128,3])
                #image = imresize(imread(image_path), [128,128, 3])
                #image = image.astype('float32')
                img[:,:,0] -= 123.68
                img[:,:,1] -= 116.78
                img[:,:,2] -= 103.94
                #image = image/255
                #image = 1-image
                Y_train.append(i)
                X_train.append(img)
                count += 1
            else:
                img = image.img_to_array(img)
                #Y.append(image)
                #image = imresize(image,[128,128,3])
                #image = imresize(imread(image_path), [128,128, 3])
                #image = image.astype('float32')
                img[:,:,0] -= 123.68
                img[:,:,1] -= 116.78
                img[:,:,2] -= 103.94
                #image = image/255
                #image = 1-image
                Y_test.append(i)
                X_test.append(img)
                count += 1
                
            print(count)
            #X.append(image.img_to_array(img))
            #X.append(image)
    return X_train,Y_train,X_test,Y_test,labels

X_train,Y_train,X_test,Y_test,labels = load_images(datasets_path)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
##X = np.array(x_train)
###print(len(x_train))
###print(x_train.shape)
##Y = np.array(y)
##print(X.shape)
##
##X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
##
##print(y_train.shape)
##print(X_train.shape)
##
number_of_classes = 101
Y_train = np_utils.to_categorical(Y_train-1, number_of_classes)
Y_test = np_utils.to_categorical(Y_test-1, number_of_classes)
gen = ImageDataGenerator(width_shift_range=.2, 
                             height_shift_range=.2,
                          zoom_range=0.2)

test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=16)
test_generator = test_gen.flow(X_test, Y_test, batch_size=16)

from keras.applications. vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

base_model = ResNet50(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
model = add_new_last_layer(base_model, 101)

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr = 1e-3,momentum = 0.9),metrics=['accuracy'])
start = time.time()
x = model.fit_generator(train_generator, epochs=5, shuffle = True,
                    validation_data=test_generator)
end = time.time()
#print(“Training time: ”,(end-start))

score = model.evaluate(X_test, Y_test)
print()
print('Test loss: ', score[0])
print('Test Accuracy', score[1])
