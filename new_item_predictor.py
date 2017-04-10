'''
Created on Mar 29, 2017

@author: Jiwoon
'''
import numpy as np
import cPickle as pickl

#from keras.models import Graph
from keras.models import model_from_json
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from operator import itemgetter
from keras.preprocessing import sequence

# Define file path
model_path = './test/ml-1m/new_item/result/1_100_200'
weight_filename = '/CNN_weights.hdf5'
model_filename = '/CNN_architecture.json'
docuPath = './test/ml-1m/new_item/document_new.all'

# Load from pretrained model
json_string = open(model_path + model_filename).read()
model = model_from_json(json_string)
model.load_weights(model_path + weight_filename)

# Predict the output of input (new-items)
D_all = pickl.load(open(docuPath, "rb")) # Load content from a file

CNN_X = D_all['X_sequence']
X_train = sequence.pad_sequences(CNN_X, maxlen=300)
Y = model.predict({'input': X_train}, batch_size=len(X_train))['output']
np.savetxt('./test/ml-1m/new_item/result/1_100_200/new_item_features.dat', Y) # Save new-item feature 