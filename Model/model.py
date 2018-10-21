import math
import os
from os import walk
import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import string
import segmentation

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Dataset/Caracteres/'
fileNames=[]
class_names_numbers = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']
class_names_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def datafetching():
    y = []
    x = []    
    for (dirpath, dirnames, filenames) in walk(TRAINING_DIR):
        fileNames.extend(filenames)
        y.append([i[0] for i in filenames]) 
        x = [(cv2.resize(cv2.imread(TRAINING_DIR + f), (25,50))[:,:,0]/255) for f in filenames]
        break
    y= np.transpose(y)
    y_nums = []
    y_char = []
    x_nums = []
    x_char = []
    for i in range(len(y)):
        if(y.item(i).isdigit()):
            y_nums.append(y[i])
            x_nums.append(x[i])
        else:
            y_char.append(y[i])
            x_char.append(x[i])
    x_nums = np.array(x_nums)
    y_nums = np.array(y_nums)
    x_char = np.array(x_char)
    y_char = [[ord(i[0])-97] for i in y_char]
    y_char = np.array(y_char)
    x_nums, y_nums = shuffle_in_unison(x_nums,y_nums)
    x_char,y_char =shuffle_in_unison(x_char,y_char)
    x_nums_training, x_nums_test = x_nums[:,:], x_nums[188:,:]
    y_nums_training, y_nums_test = y_nums[:,:], y_nums[188:,:]
    x_char_training, x_char_test = x_char[:,:], x_char[188:,:]
    y_char_training, y_char_test = y_char[:,:], y_char[188:,:]
    return x_nums_training,x_nums_test,y_nums_training,y_nums_test,x_char_training,x_char_test,y_char_training,y_char_test

    
    
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def trainNums(x_train,y_train,x_test,y_test):
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50, 25)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    return model

def trainChar(x_train,y_train,x_test,y_test):
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50,25)),
    keras.layers.Dense(128, activation=tf.nn.tanh),
    keras.layers.Dense(26, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    return model


"""plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_nums[i], cmap=plt.cm.binary)
    plt.xlabel(y_nums[i])
plt.show()"""


def main():
    x_nums_training,x_nums_test,y_nums_training,y_nums_test,x_char_training,x_char_test,y_char_training,y_char_test = datafetching()
    nummod=trainNums(x_nums_training,y_nums_training,x_nums_test,y_nums_test)
    charmod=trainChar(x_char_training,y_char_training,x_char_test,y_char_test)
    preds = segmentation.characters
    for i in preds:
        img = cv2.bitwise_not(i)
        x = np.array([cv2.resize(img, (25,50))[:,:]])        
        numpred=nummod.predict(x)
        charpred=charmod.predict(x)
        if np.max(charpred) < np.max(numpred):
            print(np.argmax(numpred, axis =1))
        else:
            print(chr(np.argmax(charpred, axis =1)+97))
if __name__ == '__main__':
    main()