## Run the Convoluted Neural Net for the complete genomes of single stranded RNA viruses

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

print('\n\nModelling the single-stranded RNA viruses:\n target = ssRNA positive\n')

# Load the padded and encoded series of sequences
print('loading data...')
X2 = np.load('X2_genome_array.npy')
y2 = np.load('y2_genome_array.npy')
print('done\n')

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.33, stratify = y2)

X2_train = X2_train.reshape(X2_train.shape[0],1,len(X2[0]),1)
X2_test = X2_test.reshape(X2_test.shape[0],1,len(X2[0]),1)

#Print a baseline (from the jupyter notebook)
print('Baseline: 69.83%')

# CNN to classify ssRNA viruses (2)
model_2 = Sequential()
model_2.add(Conv2D(50, kernel_size =(1,150), input_shape=(X2_train.shape[1:]), activation='relu'))
model_2.add(MaxPool2D((1, 5)))
model_2.add(Conv2D(40, (1,140), activation='relu'))
model_2.add(MaxPool2D((1, 4)))
model_2.add(Conv2D(30, (1,130), activation='relu'))
model_2.add(MaxPool2D((1, 3)))
model_2.add(Conv2D(20, (1,120), activation='relu'))
model_2.add(MaxPool2D((1, 2)))
model_2.add(Flatten())
model_2.add(Dense(50, activation='relu'))
model_2.add(Dense(y2.shape[1], activation='sigmoid'))

model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', metrics.binary_accuracy])

model_2.fit(X2_train, y2_train, validation_data = (X2_test, y2_test), epochs = 10)
