#!/usr/bin/env python
# coding: utf-8

# Considering I have a lot of images provided by JBM group. 
# 
# When I analysed the images given I could see that images are chosen faulty based on shorter component lengths, welding dent, cracks etc
# 
# So based on the feature set of these images and complexity of the problem, my first thought was to use neural networks.When coming to neural networks I chose CNN because we have an image dataset and CNN have shown greater capabilities when handling image datasets with respect to complexity and accuracy. 
# 
# Preprocessing:
# 
# Assuming the images I received are of x^2 pixels each and I received total N images. So after converting these images to the desired array format that my algorithm wants I will move on to passing it through various layers of my CNN like padding, max pool, hiddenlayers etc. My model will have 2 output labels ie "OK" and "FAULTY" which will be predicted based on the feature set of the images(x_train).
# 
# The assignment asked to develop a model to determine these classes correctly so below is the thought process and logic to reach to the desired result.

# In[ ]:


# Using Tensorflow to implement CNN
import tensorflow as tf 


# In[ ]:


# Assuming we have the training data flattened into x_train and y_train respectively
x_train, y_train 


# In[ ]:


# these first four lines represent the input pixels and the length width and height of our filter for CNN. These numeric values wrten below
#are just for exmaple. These values can be replaced with the actual values using JBM images. This code is just to show how to I wish to develop the model.

input_width = 28
input_height = 28
input_channels = 1
input_pixels = 784


# The below 8 lines represent the various values for stride, max pool, convolutional layer size etc. The numbers are again random and can be optimised further.
n_conv1 = 32 
n_conv2 = 64
stride_conv1 = 1 
stride_conv2 = 1 
conv1_k = 5
conv2_k = 5
max_pool1_k = 2
max_pool2_k = 2

n_hidden = 1024 #no. of hidden layers
n_out = 2 #2 output classes

#Finding input_size_to_hidden for quick use in latter part of code.

input_size_to_hidden = (input_width//(max_pool1_k*max_pool2_k)) * (input_height//(max_pool1_k*max_pool2_k)) *n_conv2


# In[ ]:


#Defining the weights and biases for each layer and initialising them with random numners using tensorflow's normal distribution random number generator.
# wc1 = weights for convolutional layer 1
# wc2 = weights for convolutional layer 2
# wh1 = weights for hidden layer
# w0 = weights for output layer


weights = {
    "wc1" : tf.Variable(tf.random_normal([conv1_k, conv1_k, input_channels, n_conv1])),
    "wc2" : tf.Variable(tf.random_normal([conv2_k, conv2_k, n_conv1, n_conv2])),
    "wh1" : tf.Variable(tf.random_normal([input_size_to_hidden, n_hidden])),
    "wo" : tf.Variable(tf.random_normal([n_hidden, n_out]))
}

# bc1 = biases for convolutional layer 1
# bc2 = biases for convolutional layer 1
# bh1 = biases for hidden layer
# b0 = biases for output layer


biases = {
    "bc1" : tf.Variable(tf.random_normal([n_conv1])),
    "bc2" : tf.Variable(tf.random_normal([n_conv2])),
    "bh1" : tf.Variable(tf.random_normal([n_hidden])),
    "bo" : tf.Variable(tf.random_normal([n_out])),
}


# In[ ]:


# Defining covolutional and max pooling logic

def conv(x, weights, bias, strides = 1):
    out = tf.nn.conv2d(x, weights, padding="SAME", strides = [1, strides, strides, 1])
    out = tf.nn.bias_add(out, bias)
    out = tf.nn.relu(out)
    return out

def maxpooling(x, k = 2):
    return tf.nn.max_pool(x, padding = "SAME", ksize = [1, k, k, 1], strides = [1, k, k, 1])


# In[ ]:


# Defining the CNN model using all the things defined above. Forward propagation basically.
def cnn(x, weights, biases, keep_prob):
    x = tf.reshape(x, shape = [-1 ,input_height, input_width, input_channels])
    conv1 = conv(x, weights['wc1'], biases['bc1'], stride_conv1)
    conv1_pool = maxpooling(conv1, max_pool1_k)
    
    conv2 = conv(conv1_pool, weights['wc2'], biases['bc2'], stride_conv2)
    conv2_pool = maxpooling(conv2, max_pool2_k)
    
    hidden_input = tf.reshape(conv2_pool, shape = [-1, input_size_to_hidden])
    hidden_output_before_activation = tf.add(tf.matmul(hidden_input, weights['wh1']), biases['bh1'])
    hidden_output_before_dropout = tf.nn.relu(hidden_output_before_activation)
    hidden_output = tf.nn.dropout(hidden_output_before_dropout, keep_prob) 
   
    output = tf.add(tf.matmul(hidden_output, weights['wo']), biases['bo'])
    return output


# In[ ]:


#Finding predictions and storing in pred

x = tf.placeholder("float", [None, input_pixels])
y = tf.placeholder(tf.int32, [None, n_out])
keep_prob = tf.placeholder("float")
pred = cnn(x, weights, biases, keep_prob)


# In[ ]:


# defining cost to be mean of all the values using all features to optimize further in the back propagation phase.

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels = y))


# In[ ]:


#Initialsing optimizer. Using AdamOptimizer in this case.
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
optimize = optimizer.minimize(cost)


# In[ ]:


#Starting tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[ ]:


# The no.of times we run this part the same no' of times the cost keeps getting reduced until reaching stagnation.
c, _ = sess.run([cost,optimize], feed_dict={x:x_train , y:y_train})
c


# In[ ]:


# To find the feature with max probability.
predictions = tf.argmax(pred, 1)


# This was my overall approach to build the model and how the data will flow. The output can now be generated for the two classes.

