## Run the data processing and Convoluted Neural Net for the complete genomes of double stranded DNA viruses with no RNA stage

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
print('\n------------------------------------------------------------------------')
print('\n\nModelling double-stranded RNA viruses:\n target = Sedoreovirinae\n')

# Load the padded and encoded series of sequences
print('loading data')
X4 = np.load('X4_genome_array.npy')
y4 = np.load('y4_genome_array.npy')
print('done\n')

# Set up the target variable and train_test_split, with an equal proportion of target variables in test/train
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = 0.33, stratify = y4)

# Reshape the input variables to make them 3D
X4_train = X4_train.reshape(X4_train.shape[0],1,len(X4[0]),1)
X4_test = X4_test.reshape(X4_test.shape[0],1,len(X4[0]),1)

#Print a baseline (from the jupyter notebook)
print('Baseline: 70.67%')

# CNN to classify dsRNA viruses (4)
model_4 = Sequential()
model_4.add(Conv2D(50, kernel_size =(1,200), input_shape=(X4_train.shape[1:]), activation='relu'))
model_4.add(MaxPool2D((1, 5)))
model_4.add(Conv2D(40, (1,200), activation='relu'))
model_4.add(MaxPool2D((1, 4)))
model_4.add(Conv2D(30, (1,200), activation='relu'))
model_4.add(MaxPool2D((1, 3)))
model_4.add(Conv2D(20, (1,200), activation='relu'))
model_4.add(MaxPool2D((1, 2)))
model_4.add(Flatten())
model_4.add(Dense(50, activation='relu'))
model_4.add(Dense(y4.shape[1], activation='sigmoid'))
print('\n',model_4.summary(),'\n')
model_4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', metrics.binary_accuracy])
ep = int(input('# of epochs? '))
model_4.fit(X4_train, y4_train, validation_data = (X4_test, y4_test), epochs = ep)
model_4.save('model_4.h5')
print('\n------------------------------------------------------------------------')
