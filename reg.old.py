import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#x_data = np.array([2,3,4,5,6],dtype=np.int32) #independent/predictor
#y_data = np.array([4,6,8,10,12],dtype=np.int32) #dependent/response

x_data = np.array([2.0,3.0,4.0,5.0,6.0],dtype=np.float32) #independent/predictor
y_data = np.array([4.0,6.0,8.0,10.0,12.0],dtype=np.float32) #dependent/response

x_test = np.array([7],dtype=np.float32)

"""tf.keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
units=1 (or 2) => dimension of output, 

input_shape =[1] => dimension of input
a shape (30,4,10) means an array or tensor with 3 dimensions, 
containing 30 elements in the first dimension, 4 in the second and 10 in the third, 
totaling 30*4*10 = 1200 elements or numbers.

if shape is 1 dimension, no need for tupple, use a scalar such as input_shape=(3,) or input_dim=3 for 3 elements

no activation=> a(x)=x =>linear

Keras skips the first param=batch size, i.e  30 images of 50x50 pixels in RGB (3 channels), the shape of your 
input data is (30,50,50,3) and you can use input_shape=(50,50,3) to handle any number of images


"""
#model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1]) ])
#model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=(1,)) ])



model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(1,))) #1 dimension input 
model.add(tf.keras.layers.Dense(units=1)) #one value output

""" model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()]) 
String (name of optimizer) or optimizer instance. See tf.keras.optimizers.

#a loss function is any callable with the signature loss = fn(y_true,y_pred), 
#where y_true are the ground truth values, and y_pred are the model's predictions. 
#y_true should have shape (batch_size, d0, .. dN) """

model.compile(optimizer='SGD', loss='MSE') #mape, MSE,mean_squared_error

tn_model = model.fit(x_data,y_data,epochs=15)

#sys.exit() #terminate at this point
#os.system('cls') #clear screen in DOS, 

y_pred = model.predict(x_test)
print((y_pred))

plt.scatter(x_data, y_data, label='Training Data')
plt.plot(x_test, y_pred, 'ro', label='Predictions')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show() 

