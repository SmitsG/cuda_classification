# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:37:43 2018

@author: xx_xx
"""

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
#%matplotlib inline

# de verschillende sets
train_path = 'train'
valid_path = 'valid'
test_path = 'test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['Positive_TB', 'Negative_TB'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['Positive_TB', 'Negative_TB'], batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['Positive_TB', 'Negative_TB'], batch_size=100)

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
        
imgs, labels = next(train_batches)

plots(imgs, titles=labels)

# Build and train CNN
model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)),
        Flatten(),
        Dense(2, activation='softmax'),
    ])

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
#steps is het aantal batches, stel 30 afbeeldingen, batch_size = 10 --> 3 steps
model.fit_generator(train_batches, steps_per_epoch=33, 
                    validation_data=valid_batches, validation_steps=16, epochs=5, verbose=2)

#Predict
test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)

test_labels = test_labels[:,0]
test_labels
predictions = model.predict_generator(test_batches, steps=1, verbose=0)
predictions

cm = confusion_matrix(test_labels, predictions[:,0])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

 #  plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm_plot_labels = ['Negative_TB','Positive_TB']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

# Build Fine-tuned VGG16 model
#een kleinere test_batch omdat mijn(Valerie) laptop het niet aan kan bij vgg16
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['Positive_TB', 'Negative_TB'], batch_size=10)


vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()

type(vgg16_model)

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
model.summary()

model.layers.pop()
model.summary()


for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation='softmax'))
model.summary()

#Train the fine-tuned VGG16 model
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=33, 
                    validation_data=valid_batches, validation_steps=16, epochs=5, verbose=2)

#old model results
#model.fit_generator(train_batches, steps_per_epoch=4, 
#                    validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

#Predict using fine-tuned VGG16 model¶
test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)

test_labels = test_labels[:,0]
test_labels
predictions = model.predict_generator(test_batches, steps=1, verbose=0)
cm = confusion_matrix(test_labels, np.round(predictions[:,0]))
cm_plot_labels = ['Negative_TB','Positive_TB']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

