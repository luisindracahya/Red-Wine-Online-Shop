#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# In[3]:


dataset = pd.read_csv('E202-COMP7117-TD01-00 - clustering.csv')
dataset['SpecialDay'] = dataset['SpecialDay'].replace({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
dataset['VisitorType'] = dataset['VisitorType'].replace({'Returning_Visitor': 2, 'New_Visitor': 1, 'Other': 0})
dataset['Weekend'] = dataset['Weekend'].replace({True: 1, False: 0})
dataset.head()


# In[4]:


feature = dataset[[
    'SpecialDay',
    'VisitorType',
    'Weekend',
    'ProductRelated_Duration',
    'ExitRates'
]]
feature.head()


# In[5]:


from sklearn.preprocessing import MinMaxScaler

x = feature.values #returns a numpy array
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature = pd.DataFrame(x_scaled)
feature.head()


# In[6]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(feature)
x = principalComponents


# In[13]:


# x.shape


# In[7]:


# features = range(pca.n_components_)
# plt.bar(features, pca.explained_variance_ratio_, color='black')
# plt.xlabel('PCA features')
# plt.ylabel('variance %')
# plt.xticks(features)


# In[8]:


class SOM:
    
    def __init__(self, height, width, input_dimension):
        self.height = height
        self.width = width
        self.input_dimension = input_dimension
        self.location = [tf.to_float([y,x]) for y in range(height) for x in range(width)]

        self.weight = tf.Variable(tf.random_normal([width*height, input_dimension]))
        self.input = tf.placeholder(tf.float32, [input_dimension])

        self.best_matching_unit = self.get_bmu()
#         print(best_matching_unit)
        self.updated_weight, self.rate_stacked = self.update_neigbour(self.best_matching_unit)
        
    def get_bmu(self):
        square_difference = tf.square(self.input - self.weight)
        distance = tf.sqrt(tf.reduce_mean(square_difference, axis = 1))

        bmu_index = tf.argmin(distance)
        # print('index : ' + str(bmu_index))
        bmu_location = tf.to_float([tf.div(bmu_index, self.width), tf.mod(bmu_index, self.width)])
        # print('0 : ' + str(bmu_location[0]))
        # print('1 : ' + str(bmu_location[1]))
        return bmu_location
    
    def update_neigbour(self, bmu):
        learning_rate = 0.1

        sigma = tf.to_float(tf.maximum(self.width, self.height)/2)

        # Calculate distance for each cluster from winning node
        square_difference = tf.square(self.location - bmu)
        distance = tf.sqrt(tf.reduce_mean(square_difference, axis = 1))

        neighbour_strength = tf.exp(tf.div(tf.negative(tf.square(distance)), 2 * tf.square(sigma)))

        rate = neighbour_strength * learning_rate
        total_node = self.width * self.height
        rate_stacked = tf.stack([tf.tile(tf.slice(rate, [i], [1]), [self.input_dimension]) for i in range(total_node)])

        input_weight_difference = tf.subtract(self.input, self.weight)

        weight_difference = tf.multiply(rate_stacked, input_weight_difference)

        weight_new = tf.add(self.weight, weight_difference)

        return tf.assign(self.weight, weight_new), rate_stacked
    
    
    def train(self, dataset, epoch):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(epoch):
                for data in x:
                    sess.run(self.updated_weight, feed_dict={self.input : data})                    
                    
            for i, data in enumerate(x):
                print('{}th data winning node: '.format(i+1), end='')
                print(sess.run(self.best_matching_unit, feed_dict={self.input : data}))

            cluster = [[] for i in range(self.height)]
            location = sess.run(self.location)
            weight = sess.run(self.weight)

            # print('location: ' + str(location))
            
            
            for i, loc in enumerate(location):
#                 print(i, loc[0])
                # print(i, loc)
                cluster[int(loc[0])].append(weight[i])

        
            self.cluster = cluster


# In[9]:


def main():
    x = principalComponents
    height = 3
    width = 3
    input_dimension = 3
    epoch = 5000
    
    som = SOM(height, width, input_dimension)
    
    som.train(x, epoch)
    
    plt.imshow(som.cluster)
    plt.show()
    
    
main()


