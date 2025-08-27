#Tensor Basics - https://www.guru99.com/tensor-tensorflow.html
import tensorflow as tf

#create constants 
#rank 0 = scalars
r1 = tf.constant(1, tf.int32)
print(r1)

input("Press ENTER to continue")

r2 = tf.constant(1, tf.int32, name="my_scalar")
print(r2)

#rank 1, i.e. 1-dimension tensors
r1_v = tf.constant([1,3,5], tf.int32)
print(r1_v)
r2_v = tf.constant([True, False, True, True], tf.bool)
print(r2_v)
input("Press ENTER to continue")

#rank 2 i.e 2D tensors
r2_mat = tf.constant([ [1,2], [3,4] ], tf.int32)
print(r2_mat)

#rank 3
r3_mat = tf.constant([ [1,2], [3,4], [5,6] ], tf.int32)
print(r3_mat)
print(r3_mat.shape) #extract the shape of a tensor

#create a vector of zeros, i.e rank 1 tensor
print( tf.zeros(5))
input("Press ENTER to continue")

# Create a vector of 1
print(tf.ones([10, 10]))

# Create a vector of ones with the same number of rows as r3_mat
print(tf.ones(r3_mat.shape[0]))
input("Press ENTER to continue")

# Create a vector of ones with the same number of columns as r3_mat
print(tf.ones(r3_mat.shape[1]))

#create 3x2 tensor with ones
print(tf.ones(r3_mat.shape))

#casting, i.e. Change type of data
type_float = tf.constant(3.123456789, tf.float32)
type_int = tf.cast(type_float, dtype=tf.int32)
print(type_float.dtype)
print(type_int.dtype)

#operations: tf.add(a, b), tf.substract(a, b), tf.multiply(a, b), tf.div(a, b),  tf.pow(a, b), tf.exp(a), tf.sqrt(a)
x = tf.constant([2.0], dtype = tf.float32)
print(tf.sqrt(x))

# Add
tensor_a = tf.constant([[1,2]], dtype = tf.int32)
tensor_b = tf.constant([[3, 4]], dtype = tf.int32)

tensor_add = tf.add(tensor_a, tensor_b)
print(tensor_add)

# Multiply
tensor_multiply = tf.multiply(tensor_a, tensor_b)
print(tensor_multiply)

var = tf.Variable(name="myVar", initial_value=[1,2], type_int=tf.int32)
print(var)
input("Press ENTER to continue")

my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)
print(my_variable)
input("Press ENTER to continue")

# Variables can be all kinds of types, just like tensors
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])

print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As NumPy: ", my_variable.numpy())
input("Press ENTER to continue")

print("Shape: ", complex_variable.shape)
print("DType: ", complex_variable.dtype)
print("As NumPy: ", complex_variable.numpy())

input("Press ENTER to continue")
print("A variable:", my_variable)
print("\nViewed as a tensor:", tf.convert_to_tensor(my_variable))
print("\nIndex of highest value:", tf.math.argmax(my_variable))

# This creates a new tensor; it does not reshape the variable.
print("\nCopying and reshaping: ", tf.reshape(my_variable, [1,4]))
print("A variable:", my_variable) #original variable not reshaped
input("Press ENTER to continue")

# TensorFlow works around 3 main components: Graph, Tensor, and Session
#graph: all math opereations happens inside a graph, nodes are ops, and the graph is a project where ops are done
#Tensor representation of data progressing between  ops
#session: a session will execute the ops from the graph. A session need to be open so the the values of a tesnsor
#can be fed into a graph. Inside the session, opeator need to be executed to create an output. Graph and session are independent,
# a session could be executed to get the input for use in a graph later

"""Example:Create two tensors
Create an operation
Open a session
Print the result"""

## Create, run  and evaluate a session
x = tf.constant([2])
y = tf.constant([4])

## Create operator
multiply = tf.multiply(x, y)

## Create a session to run the code
# TF2 use earger execution, session is implied
#sess = tf.compat.v1.Session()
#result_1 = sess.run(multiply) 
print(multiply)
#sess.close()
input("Press ENTER to continue")

# graph
#a = tf.Variable("a", dtype=tf.int32, initial_value= tf.constant([5]))
#b = tf.Variable("b", dtype=tf.int32,  initial_value=tf.constant([6]))
a = tf.Variable(name="a", initial_value=[5], type_int=tf.int32)
b = tf.Variable(name="b", initial_value=[6], type_int=tf.int32)
c = tf.constant([5], name =	"constant")
square = tf.constant([2], name =	"square")
f = tf.multiply(a, b) + tf.pow(a, square) + b + c
print(f)


