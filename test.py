import keras
import os
from keras.models import Sequential, Model, Input
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications import imagenet_utils
from keras.models import load_model
from typing import Tuple, List
import glob
import pandas as pd
import numpy as np
from skimage import feature
# start tensorflow interactiveSession
import tensorflow as tf
from PIL import Image
from keras import applications
np.random.seed(10)
# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import spatial
import pickle
import argparse
import sys


parser = argparse.ArgumentParser(description="Testing the image")
parser.add_argument('testing image path', metavar='imagepath', help='path of testing image path')
parser.add_argument('testing directory', metavar='val_dir', help='path of testing data directory')
args = parser.parse_args()
imagepath = sys.argv[1]
val_dir = sys.argv[2]



## load the models
model_vgg = VGG16(weights = 'model/vgg16_weights.hdf5',include_top=False)
model_V3 = InceptionV3(weights="model/inception_v3.hdf5", include_top=False)
model_cnn = load_model("model/cnn_model.best.hdf5")
model_cnn1 = Sequential()
for layer in model_cnn.layers[:-3]: # go through until last three layer
    model_cnn1.add(layer)
pca_reload = pickle.load(open("model/pca_mode.pkl",'rb'))
loaded_model = pickle.load(open('model/logistic_model.sav', 'rb'))


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
    return features_flatten
def create_lbp_features(dataset):
    
    x_scratch = []
    # loop over the images
    for imagePath in dataset:
        image = load_img(imagePath,color_mode="grayscale")
        desc = LocalBinaryPatterns(24, 8)
        hist = desc.describe(image)
        x_scratch.append(hist)        
    x = np.vstack(x_scratch)
    return x



## make the dict
dict = {}
k = 0
for i in sorted(os.listdir(val_dir)):
    dict[k] = i
    k = k+1

## testing
test = [imagepath]
vgg_test_features_flatten = create_vgg_features(test, model_vgg)
v3_test_features_flatten = create_v3_features(test, model_V3)
cnn_test_features_flatten = create_cnn_feature(test, model_cnn1)
test__lbp_features_flatten = create_lbp_features(test)
t = np.append(vgg_test_features_flatten,v3_test_features_flatten,axis=1)
t = np.append(t,cnn_test_features_flatten,axis=1)
t = np.append(t,test__lbp_features_flatten,axis=1)
t = pca_reload.transform(t)
test_pred = loaded_model.predict(t[0:2])
print("predicted class:"+dict[test_pred[0]])


## zero-shot learning
print("### nearest five points in traingin dataset ###")
embedding = np.load('model/embedding.npy')
label = np.load('model/label.npy')
index_x = spatial.KDTree(embedding).query(t[0],k=5)[1]
for i in index_x:
    print(dict[label[i]])