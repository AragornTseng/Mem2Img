import keras
import os
from keras.models import Sequential, Model, Input
from keras import regularizers
from keras.callbacks import History
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.engine import training
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout, Activation, Average, Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16, imagenet_utils
from keras.applications.inception_v3 import InceptionV3
from typing import Tuple, List
import glob
import pandas as pd
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
# start tensorflow interactiveSession
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras import applications
from keras.models import load_model
np.random.seed(10)
# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import itertools
import gc
import pickle
import sys
import argparse


parser = argparse.ArgumentParser(description="Training the model")
parser.add_argument('training directory', metavar='train_dir', help='path of training data directory')
parser.add_argument('testing directory', metavar='val_dir', help='path of testing data directory')
args = parser.parse_args()
train_dir = sys.argv[1]
val_dir = sys.argv[2]



## preprocessing
train = []
train_y = []
val = []
val_y = []
label_train = -1
label_test = -1
for i in sorted(os.listdir(train_dir)):
  label_train+=1
  for img in sorted(os.listdir(train_dir+'/'+i)):
    train.append(os.path.join(train_dir,i,img))
    train_y.append(label_train)
for i in sorted(os.listdir(val_dir)):
  label_test+=1
  for img in sorted(os.listdir(val_dir+'/'+i)):
    val.append(os.path.join(val_dir,i,img))
    val_y.append(label_test)
num_classes = len(os.listdir(train_dir))
y_train = np_utils.to_categorical(train_y, num_classes)
y_val = np_utils.to_categorical(val_y, num_classes)

## define the model
model_vgg16 = VGG16(weights="imagenet", include_top=False)
model_V3 = InceptionV3(weights="imagenet", include_top=False)

model_cnn = Sequential()
model_cnn.add(Conv2D(32, (3, 3), activation='relu',input_shape = (224,224,3)))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Conv2D(128, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(GlobalAveragePooling2D())
model_cnn.add(Dense(64, activation='relu'))
model_cnn.add(Dense(num_classes, activation='softmax'))


# create data for training and testing

def create_train(dataset, pre_model):
    x_scratch = []
    # loop over the images
    for imagePath in dataset:
        image = load_img(imagePath, target_size=(224, 224),color_mode="rgb")
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        x_scratch.append(image)       
    x = np.vstack(x_scratch)
    return x

train_x = create_train(train, model_cnn)
val_x = create_train(val, model_cnn)

## generate weight
weights = {}
k_weights = 0
for i in sorted(os.listdir(train_dir)):
    count = len(os.listdir(os.path.join(train_dir,i)))
    weight = train_x.shape[0]/(num_classes*count)
    weights[k_weights] = weight
    k_weights = k_weights+1


# Creating a checkpointer 
checkpointer = ModelCheckpoint(filepath='cnn_model.best.hdf5', 
                               verbose=1,save_best_only=True)
model_cnn.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

#Fitting the model on the train data and labels.
history = model_cnn.fit(train_x, y_train, 
          batch_size=32, epochs=32, class_weight=weights,
          verbose=1, callbacks=[checkpointer], 
          validation_data=(val_x, y_val), shuffle=True)


model_cnn1 = Sequential()
for layer in model_cnn.layers[:-3]: # go through until last layer
    model_cnn1.add(layer)


## Create feature
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist
def create_vgg_features(dataset, pre_model):
    
    x_scratch = []
    from keras.applications.vgg16 import preprocess_input
    for imagePath in dataset:
        image = load_img(imagePath, target_size=(224, 224),color_mode="rgb")
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        x_scratch.append(image)        
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 7 * 7 * 512))
    del x, features
    gc.collect()
    return features_flatten
def create_v3_features(dataset, pre_model):
    
    x_scratch = []
    from keras.applications.inception_v3 import preprocess_input
    for imagePath in dataset:
        image = load_img(imagePath, target_size=(224, 224),color_mode="rgb")
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        x_scratch.append(image)      
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 5 * 5 * 2048))
    del x, features
    gc.collect()
    return features_flatten
def create_cnn_feature(dataset, pre_model):
    
    x_scratch = []
    for imagePath in dataset:
        image = load_img(imagePath, target_size=(224, 224),color_mode="rgb")
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        x_scratch.append(image)        
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 12 * 12 * 128))
    del x, features
    gc.collect()
    return features_flatten
def create_lbp_features(dataset):
    
    x_scratch = []
    # loop over the images
    for imagePath in dataset:
        image = load_img(imagePath,color_mode="grayscale")
        desc = LocalBinaryPatterns(24, 8)
        #desc = LocalBinaryPatterns(16, 2)
        hist = desc.describe(image)
        x_scratch.append(hist)        
    x = np.vstack(x_scratch)
    return x

vgg_train_features_flatten = create_vgg_features(train, model_vgg16)
vgg_val_features_flatten = create_vgg_features(val, model_vgg16)
v3_train_features_flatten = create_v3_features(train, model_V3)
v3_val_features_flatten = create_v3_features(val, model_V3)
cnn_features_flatten = create_cnn_feature(train, model_cnn1)
cnn_val_features_flatten = create_cnn_feature(val, model_cnn1)
train_lbp_features_flatten = create_lbp_features(train)
val__lbp_features_flatten = create_lbp_features(val)
x = np.append(vgg_train_features_flatten,v3_train_features_flatten,axis=1)
x = np.append(x,cnn_features_flatten,axis=1)
x = np.append(x,train_lbp_features_flatten,axis=1)
y = np.append(vgg_val_features_flatten,v3_val_features_flatten,axis=1)
y = np.append(y,cnn_val_features_flatten,axis=1)
y = np.append(y,val__lbp_features_flatten,axis=1)



## Training
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(x)

# Apply transform to both the training set and the test set.
x = scaler.transform(x)
y = scaler.transform(y)

pca = PCA(.95)

pca.fit(x)
x = pca.transform(x)
y = pca.transform(y)
logisticRegr = LogisticRegression(solver = 'lbfgs',class_weight= weights)
logisticRegr.fit(x, train_y)
score = logisticRegr.score(y, val_y)
print("Testing Score:"+score)
pickle.dump(pca, open("pca_mode.pkl","wb"))
pickle.dump(logisticRegr, open('logistic_model.sav', 'wb'))