## Run the Convoluted Neural Net for the 4 main classifications

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
print('\n\nModelling the 4 most common virus classifications in the dataset:\n ssRNA, dsRNA, ssDNA, and dsDNA with no RNA stage\n')

# Load the padded and encoded series of sequences
print('loading data...')
X0 = np.load('X0_genome_array.npy')
y0 = np.load('y0_genome_array.npy')
print('done\nsplitting & reshaping...')

# Set up the target variable and train_test_split, with an equal proportion of target variables in test/train
X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size = 0.33, stratify = y0)

# Reshape the input variables to make them 3D
X0_train = X0_train.reshape(X0_train.shape[0],1,len(X0[0]),1)
X0_test = X0_test.reshape(X0_test.shape[0],1,len(X0[0]),1)

##Print a baseline (from the jupyter notebook)
print('Baseline: 27.47%')

# CNN to classify ssDNA viruses (1)
model_0 = Sequential()
model_0.add(Conv2D(50, kernel_size =(1,100), input_shape=(X0_train.shape[1:]), activation='relu'))
model_0.add(MaxPool2D((1, 5)))
model_0.add(Conv2D(40, (1,100), activation='relu'))
model_0.add(MaxPool2D((1, 4)))
model_0.add(Conv2D(30, (1,100), activation='relu'))
model_0.add(MaxPool2D((1, 3)))
model_0.add(Conv2D(20, (1,100), activation='relu'))
model_0.add(MaxPool2D((1, 2)))
model_0.add(Flatten())
model_0.add(Dense(50, activation='relu'))
model_0.add(Dense(y0.shape[1], activation='softmax'))
print('\n',model_0.summary(),'\n')
model_0.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', metrics.categorical_accuracy])
ep = int(input('# of epochs? '))
model_0.fit(X0_train, y0_train, validation_data = (X0_test, y0_test), epochs = ep)
plot_model(model_0, to_file='model_0_plot.png', show_shapes=True, show_layer_names=True)
model_0.save('model_0.h5')
print('\n---------------------------------------------------------------------------------------------------\n')
