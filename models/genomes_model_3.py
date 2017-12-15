## Run the Convoluted Neural Net for the complete genomes of double stranded DNA viruses with no RNA stage

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
print('\n-----------------------------------------------------------------------------------------------------')
print('\n\nModelling double-stranded DNA with no RNA stage viruses:\n target = Siphoviridae\n')

# Load the padded and encoded series of sequences
print('loading data...')
X3 = np.load('X3_genome_array.npy')
y3 = np.load('y3_genome_array.npy')
print('done\nsplitting & reshaping...\n')

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = 0.33, stratify = y3)

X3_train = X3_train.reshape(X3_train.shape[0], 1, len(X3[0]), 1)
X3_test = X3_test.reshape(X3_test.shape[0], 1, len(X3[0]), 1)

#Print a baseline (from the jupyter notebook)
print('Baseline: 62.15%')

# Build and run the CNN
model_3 = Sequential()
model_3.add(Conv2D(50, kernel_size =(1,100), input_shape=(X3_train.shape[1:]), activation='relu'))
model_3.add(MaxPool2D((1, 5)))
model_3.add(Conv2D(40, (1,100), activation='relu'))
model_3.add(MaxPool2D((1, 4)))
model_3.add(Conv2D(30, (1,100), activation='relu'))
model_3.add(MaxPool2D((1, 3)))
model_3.add(Conv2D(30, (1,100), activation='relu'))
model_3.add(MaxPool2D((1, 3)))
model_3.add(Conv2D(20, (1,100), activation='relu'))
model_3.add(MaxPool2D((1, 2)))
model_3.add(Flatten())
model_3.add(Dense(50, activation='relu'))
model_3.add(Dense(y3.shape[1], activation='sigmoid'))

model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', metrics.binary_accuracy])

model_3.fit(X3_train, y3_train, validation_data = (X3_test, y3_test), epochs = 10)

print('-----------------------------------------------------------------------------------------------------\n')
