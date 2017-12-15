# Modelling complete genomes of single stranded DNA viruses

# imports
import pandas as pd
import numpy as np
import progressbar
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Reshape
from keras.utils import to_categorical, plot_model
from keras import metrics
import warnings
# Don't show warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings('ignore')

print('\n\nModelling complete genomes of single stranded DNA viruses:\n target = Geminiviridae\n')

# Load the padded and encoded series of sequences
print('loading data...')
X1 = np.load('X1_genome_array.npy')
y1 = np.load('y1_genome_array.npy')
print('done\n')

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.33, stratify = y1)

# Reshape the input variables to make them 3D
X1_train = X1_train.reshape(X1_train.shape[0],1,len(X1[0]),1)
X1_test = X1_test.reshape(X1_test.shape[0],1,len(X1[0]),1)

#Print a baseline (from the jupyter notebook)
print('Baseline: 50.37%')

# CNN to classify ssDNA viruses (1)
model_1 = Sequential()
model_1.add(Conv2D(50, kernel_size =(1,100), input_shape=(X1_train.shape[1:]), activation='relu'))
model_1.add(MaxPool2D((1, 5)))
model_1.add(Conv2D(40, (1,100), activation='relu'))
model_1.add(MaxPool2D((1, 4)))
model_1.add(Conv2D(30, (1,100), activation='relu'))
model_1.add(MaxPool2D((1, 3)))
model_1.add(Conv2D(20, (1,100), activation='relu'))
model_1.add(MaxPool2D((1, 2)))
model_1.add(Flatten())
model_1.add(Dense(50, activation='relu'))
model_1.add(Dense(y1.shape[1], activation='sigmoid'))

model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', metrics.binary_accuracy])

model_1.fit(X1_train, y1_train, validation_data = (X1_test, y1_test), epochs = 10)
