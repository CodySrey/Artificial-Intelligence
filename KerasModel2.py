from keras.models import Model
from keras.models import Sequential 
from keras.layers import *  

####Sequential model
model = Sequential()    

#start from the first hidden layer, since the input is not actually a layer   
#but inform the shape of the input, with 3 elements.    
model.add(Dense(units=4,input_shape=(3,))) #hidden layer 1 with input

#further layers:    
model.add(Dense(units=4)) #hidden layer 2
model.add(Dense(units=1)) #output layer   
 
model.summary()
input("Press Enter to continue")

#The same can be done via functional API
#Start defining the input tensor:
inpTensor = Input((3,))   

#create the layers and pass them the input tensor to get the output tensor:    
hidden1Out = Dense(units=4)(inpTensor)  
hidden2Out = Dense(units=4)(hidden1Out)    
finalOut = Dense(units=1)(hidden2Out)   


#define the model's start and end points    
model = Model(inpTensor,finalOut)
model.summary()
 

