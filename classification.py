# Project - Artificial Neural Network
# Stephen Leonardo - 2201788634
# Luis Indracahya - 2201758934
# Kelas BD01

#import library
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# define function to load the dataset
def getDataset(filename):
    # read the dataset set given the directory and filename
    df = pd.read_csv(filename)
    
    # drop row with empty data
    df = df.dropna()
    
    # divide the dataset into feature and target
    feature = df[[
        'volatile acidity',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol'
    ]]
    target = df[['quality']]

    return feature, target

# function to derive the feature from the actual data based on the derivation formula
def derive(feature, column, value, newValue):
    for i in feature.index:
        if feature[column][i] == value[0]:
            feature[column][i] = newValue[0]
        elif feature[column][i] == value[1]:
            feature[column][i] = newValue[1]
        elif feature[column][i] == value[2]:
            feature[column][i] = newValue[2]
        else:
            feature[column][i] = newValue[3]
            
    return feature

# feed forward function using sigmoid activation function
def feed_forward():
    y1 = tf.matmul(feature_input, weight['input-hidden'])
    y1 += bias['input-hidden']
    y1 = tf.sigmoid(y1)

    y2 = tf.matmul(y1, weight['hidden-output'])
    y2 += bias['hidden-output']

    return tf.sigmoid(y2)
    
#dataset filename
filename = 'classification.csv'

#load dataset into feature and target
feature, target = getDataset(filename)    

# derive the free sulfur dioxide feature from the actual value using the derivation formula
freeSulfurDioxideValue = ['High', 'Medium', 'Low']
freeSulfurDioxideNewValue = [3, 2, 1, 0]
feature = derive(feature, 'free sulfur dioxide', freeSulfurDioxideValue, freeSulfurDioxideNewValue)

# derive the density feature from the actual value using the derivation formula
densityValue = ['Very High', 'High', 'Medium']
densityNewValue = [0, 3, 2, 1]
feature = derive(feature, 'density', densityValue, densityNewValue)

# derive the pH feature from the actual value using the derivation formula
pHValue = ['Very Basic', 'Normal', 'Very Acidic']
pHNewValue = [3, 2, 1, 0]
feature = derive(feature, 'pH', pHValue, pHNewValue)

# normalize data
scaler = MinMaxScaler()
feature = scaler.fit_transform(feature)

# encode categorical data
encoder = OneHotEncoder(sparse=False)
target = encoder.fit_transform(target)

# analyze the data with PCA to obtain the new components (highest 4 principal components) as the input of the neural network
pca = PCA(n_components=4)
feature = pca.fit_transform(feature)

# define the nodes for input, hidden, and output layer
layer = {
    'input': 4, # 4 highest components from the result of the PCA as the feature
    'hidden': 10, # hidden layer is required because the case is not linearly separable
    'output': 5 # 5 classes for the overall quality of the wine (“Great”, “Good”, “Fine”, “Decent”, and “Fair”)
}

# initialize weight randomly
weight = {
    'input-hidden': tf.Variable(tf.random_normal([layer['input'], layer['hidden']])),
    'hidden-output':  tf.Variable(tf.random_normal([layer['hidden'], layer['output']]))
}

# initialize bias randomly
bias = {
    'input-hidden': tf.Variable(tf.random_normal([layer['hidden']])),
    'hidden-output':  tf.Variable(tf.random_normal([layer['output']]))
}

# split the dataset into training(70%), validation(20%), and testing (10%)
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, train_size=0.9)
feature_train, feature_validation, target_train, target_validation = train_test_split(feature, target, train_size=0.78)

# feed data into the placeholder
feature_input = tf.placeholder(tf.float32, [None, layer['input']])
target_input = tf.placeholder(tf.float32, [None, layer['output']])

# define the learning rate
lr = 0.1

# define epoch
epoch = 5000

# define error result using mean squared error
target_predict = feed_forward()
loss = tf.reduce_mean(.5 * (target_input - target_predict) ** 2)

# define gradient descent as the optimization formula to update weight
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss)

# define saver and save path for saving model
saver = tf.train.Saver()
save_path = "saved_model/tmp/model.ckpt"

# activate tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # declare train dictionary
    train_dict = {
        feature_input: feature_train,
        target_input: target_train
    }
    
    # declare validation dictionary
    validation_dict = {
        feature_input: feature_validation,
        target_input: target_validation
    }

    # training iteration
    for i in range(1, epoch + 1):
        sess.run(train, feed_dict = train_dict)
        
        # train data
        if i % 100 == 0:
            loss_result = sess.run(loss, feed_dict = train_dict)
            
            # calculate and print error each 100 iteration
            print("Epoch Number: {}, Current Error: {}".format(i, loss_result))
            
        if i % 500 == 0:
            loss_result_validation = sess.run(loss, feed_dict = validation_dict)
            if i == 500:
                saver.save(sess, save_path)
            else:
                if loss_result_validation < loss_result_validation_prev:
                    saver.save(sess, save_path)
            loss_result_validation_prev = loss_result_validation
            
    # define the accuracy 
    match = tf.equal(tf.argmax(target_input, axis=1), tf.argmax(target_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(match, tf.float32))
    
    # declare the test dictionary
    test_dict = {
        feature_input: feature_test,
        target_input: target_test
    }
    
    # print the testing accuracy
    print("Accuracy : {}%".format(sess.run(accuracy, feed_dict = test_dict)*100))
