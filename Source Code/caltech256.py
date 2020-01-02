# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:27:18 2019

@author: Rahul
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tarfile
#import cv2
import pickle
import time
from keras.preprocessing import image

#root path is /content/drive/My Drive/caltech256
datasets_path = './256_ObjectCategories' #Add the path to the unzipped folder

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
y_train = np.array(Y_train)
X_test = np.array(X_test)
y_test = np.array(Y_test)
img_size =  128
X_train = X_train.reshape(-1,img_size,img_size,3)

X_test = X_test.reshape(-1,img_size,img_size,3)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPool2D, AvgPool2D,Flatten,GlobalAveragePooling2D

batch_size = 64

time.sleep(100)

from keras.applications.inception_resnet_v2 import InceptionResNetV2
base_model = InceptionResNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(257, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(generator.flow(X_train, y_train.values, batch_size=batch_size),len(X_train) / batch_size, epochs=20,verbose=1,validation_data=(X_test, y_test.values))

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

time.sleep(100)

from keras.optimizers import SGD
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
start = time.time()
model.fit_generator(generator.flow(X_train, y_train.values, batch_size=batch_size),len(X_train) / batch_size, epochs=30,verbose=1,validation_data=(X_test, y_test.values))
end = time.time()
print("Training time: ",(end-start))
from keras.models import load_model
model.save('inceptionresnet_model.h5') 
del model  
model = load_model('inceptionresnet_model.h5')

score = model.evaluate(x = X_test, y = y_test.values)

print("Accuracy: ",score[1],"%")
