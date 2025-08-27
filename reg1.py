import tensorflow as tf
import numpy as np
import os

x_data = np.array([2,3,4,5,6],dtype=np.int32)
y_data = np.array([4,7,10,13,16],dtype=np.int32)
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1]) ])

model.summary()