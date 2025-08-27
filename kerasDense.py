#https://www.guru99.com/keras-tutorial.html

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt 

#create data points 
# get evenly spaced numbers over a specified interval
#numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
x = data = np.linspace(1,2,20) 
y = x*4 + np.random.randn(*x.shape) * 0.3
print(x.shape)
print(*x.shape)
print(x)
print(y)
print(y.shape)
input("Enter to continue")

model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics=['mse']) #use Stochatic Gradient Descent and Mean Squared Error loss

weights = model.layers[0].get_weights()
print(weights)
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init)) 
input("Enter to continue")

model.fit(x,y, batch_size=4, epochs=50, shuffle=False)

weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))
input("Enter to continue")

predict = model.predict(data)
print(predict)

plt.plot(data, predict, 'b', data , y, 'k.')
plt.show()