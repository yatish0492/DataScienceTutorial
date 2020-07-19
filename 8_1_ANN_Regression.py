'''
ANN is 'deep learning' ML. We can use ANN to do regression and classification.

What advantage does 'ANN' regression and classification have over normal regressoion and classification?
??????????????????????

NOTE: we are creating fully connected activation layers in this program.

'''

# Importing Libraries
# --------------------
import pandas as pd
import numpy as np
import tensorflow as tf

# Importing the data
# ------------------
# The data have information about 'AT(Ambient Temperature)', 'V(Exhaust Vaccum)', 'AP(Ambient Pressure)',
# 'RH(Humidity)', 'PE(Energy output)'. 'PE' is the output with input parameter AT,V,AP,RH.
#
#   NOTE: we are reading .xlsx file so we use 'read_excel' instead of 'read_csv'
dataset = pd.read_excel('/Users/ycs/PycharmProjects/DataScieneTutorial/Data/8_1_ANN_Regression.xlsx')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Splitting data to training and test split
# -----------------------------------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# ANN has multiple layers. For creating the layers, we make use of 'Sequential()' to add the layers in a sequence.
#
ann = tf.keras.models.Sequential();

# adding input layer and first hidden layer
# -----------------------------------------
# We use 'Dense' class object which creates a layer. 
# The parameters of 'Dense' class constructor explained below,
#   units --> Number of neurons. there is no rule of thumb to decide number of neurons. it is based on experimentation and 
#                 arrive at a number which gives more accuracy.
#   activation --> activation function. we are creating 'fully connected neural network' this uses 'rectifier' activation
#                   function. 'relu' is the code name for 'rectifier' activation function.

#   NOTE : we need to create only the hidden layers, As 'Dense' will automatically create input layer when first hidden 
#            layer is added/created.
#   NOTE: The input layer neurons are basically all features/columns there in the data used for training/testing ML.
#
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu' ))


# adding 2nd hidden layer
# -----------------------
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu' )) # same code as first hidden layer.


# adding Output layer
# -------------------
# the 'units' parameter for output layer is calculated by number of outcomes we expect. in this case we are expecting 
# whether the customer exit the bank or not. that means we are expecting outcomes 'yes' or 'no' which are 2 values.
# we can output this using only 1 neuron as it will give '1' if 'yes' and '0' if 'no'. Consider if we have 3 kinds of 
# outcomes like say 'apple', 'orange' and 'pineaple' then we need 3 neurons as 1 neuron give '1' if its apple and '0' 
# if its not. 2nd neuron will give '1' if its 'orange' and '0' if its not. 3rd neuron if its 'pineaple' and '0' if its not.
#
# NOTE: always we use 'sigmoid' activation function for output layer in case of classification having only 2 categories. If
#       we have more than 2 categories to classify then we use 'softmax' activation function.
#       In case of regression then we don't specify any activation function. default activation value is 'None'
# 
ann.add(tf.keras.layers.Dense(units = 1))


# compiling ANN
# -------------
# We know that ANN will have weights on each line connecting neurons of different layers. we do use random weights first
# and the compare the predicted output with actual and calculate the error percentage and comback and re-adjust the weights
# and again do the same iteration untill we have minimum error percentage. basically we use 'Stochastic gradient descent' here.
#
# 'ann.compile()' is the method which will compile this with the above mentioned process, this has following parameters,
#       'optimizer' --> this tells which optimizer to use for doing the above mentioned process. 'adam' is the best optimizer for 
#                       'Stochastic gradient descent'
#       'loss'      --> this tells about how to calculate the error percentage between actual and predicted result. 'mean_squared_error'
#                       it means if we take multiple inputs in a batch then it is 'sum of the squired differences of each output'
#       
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')


# fitting ANN (Training)
# ----------------------
# 'ann.fit()' is the method used to fit. it accepts following parameters,
#       'batch_size' --> number of inputs in 'x_train' to be considered as one batch and execute the 'Stochastic gradient descent'. even
#                        if we don't specify this value also the default value is '32'
#       'epochs'     --> Number of iterations of checking error percentage and re-calculating the wieights and checking error percentage.
#                        basically the iterations of 'Stochastic gradient descent' in this example.
#
#   once you execute below line, in the output console it prints the 'loss' value for each iteration. after 42 iteration the 'loss' value
#       reduces to 26. and after that it keeps toggling between 26 and 27. so we can replace 'epochs' value from 100 to 42.
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)


# predicting using ANN
# --------------------
y_predicted = ann.predict(x_test)


