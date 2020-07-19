'''
ANN is 'deep learning' ML

NOTE: we are creating fully connected activation layers in this program.

What types of of neural networks are there?
    there are 2 types of neural networks,
    1) fully connected neural network
    2) convolutional neural network

    We have 2 types 'fully connected neural network' and 'convolutional neural network', in 'fully connected
    neural network' each neuron of a layer is connected every neuron of the previous layer and whereas in 
    'convolutional neural network' each neuron in a layer is not connected to all neurons of the previous 
    layer basically it connects to subset of them by making assumptions for non-connected neurons in the
    previous layer. For images n all the neurons will be more as the pixels are more so its better to go
    for 'convolutional neural network' as it avoid connections between every neurons by making assumptions
    and avoid intense computational requirement.

'''

# Importing Libraries
# --------------------
import pandas as pd
import numpy as np
import tensorflow as tf


# Importing the data
# ------------------
# The data have information about bank customer like ther name, age, gender, have_credit_card, 
# estimated slary, credit_score and whether they exited the bank or not, which we want to predict.
dataset = pd.read_csv('/Users/ycs/PycharmProjects/DataScieneTutorial/Data/8_2_ANN_Classification.csv')
x = dataset.iloc[:, 3 : -1].values # taking from 3rd column as the the names and all are useless for ML.
y = dataset.iloc[:, -1].values


# Label Encoding
# --------------
# dataset have 'Gender' column which has Male/Female which needs to be label encoded to numbers.
from sklearn.preprocessing import LabelEncoder
LabelEncoder = LabelEncoder()
x[:,2] = LabelEncoder.fit_transform(x[:,2]) # we can use 'make_column_transformer' here also. instead of 
                                            # directly calling 'LabelEncoder.fit_transform()'


# One Hot Encoding
# -----------------
# dataset have 'Geography' column which has countries which needs to be label encoded to numbers.
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
columnTransformer = make_column_transformer(
        ( OneHotEncoder(), [1]),
        remainder="passthrough"
    )
x = columnTransformer.fit_transform(x)


# Splitting data to training and test split
# -----------------------------------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Feature Scaling
# ---------------
# Feature scaling is cumpulsory in case of deep learning. We apply feature scale everything irrespective of whether if 
# the data/column have only 1 and 0 also.
# NOTE: we also could have done this feature scaling on 'x' itself before 'train_test_split'. 
from sklearn.preprocessing import StandardScaler
standardScalar = StandardScaler()
x_train = standardScalar.fit_transform(x_train)
x_test = standardScalar.transform(x_test)   # as 'fit' is already called on all columns used in 'x_test' in above line 
                                            #   itself so we directly call 'transform()'


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
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid' ))


# Compiling ANN
# -------------
# We know that ANN will have weights on each line connecting neurons of different layers. we do use random weights first
# and the compare the predicted output with actual and calculate the error percentage and comback and re-adjust the weights
# and again do the same iteration untill we have minimum error percentage. basically we use 'Stochastic gradient descent' here.
#
# 'ann.compile()' is the method which will compile this with the above mentioned process, this has following parameters,
#       'optimizer' --> this tells which optimizer to use for doing the above mentioned process. 'adam' is the best optimizer for 
#                       'Stochastic gradient descent'
#       'loss'      --> this tells about how to calculate the error percentage between actual and predicted result.
#                       if we are classigying only 2 categories i.e. binary classification then we use 'binary_crossentropy'
#                       if we are classifying more than 2 categories like 'apple', 'lemon', 'orange' then we use 'categorical_crossentropy'
#       'metrics'   --> this tells about which metrics should be considered for evaluating 'ann' during training. we have multiple metrics
#                       that can be considered but let's consider only one in this case 'accuracy'
#       
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# fitting ANN (Training)
# ----------------------
# 'ann.fit()' is the method used to fit. it accepts following parameters,
#       'batch_size' --> number of inputs in 'x_train' to be considered as one batch and execute the 'Stochastic gradient descent'. even
#                        if we don't specify this value also the default value is '32'
#       'epochs'     --> Number of iterations of checking error percentage and re-calculating the wieights and checking error percentage.
#                        basically the iterations of 'Stochastic gradient descent' in this example.
#
#   once you execute below line, in the output console it prints the 'loss' value and 'accuracy' value as we mentioned 'accuracy' in 'metrics'
#       in 'ann.compile()' for each iteration. after 20 iteration the 'accuracy' value remains almost same at 86% and 'loss' value
#       reduces to 33%. and after that it remains same. so we can replace 'epochs' value from 100 to 20.
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)


# predicting using ANN
# --------------------
y_predicted_sigmoid_probabilities = ann.predict(x_test)  # since we have used 'sigmoid' activation function. the predicted result is given in
                                                         # probabilites which is like '0.8' for 80% etc. we don't get '0' or '1'

y_predicted_binaryOutput = (y_predicted_sigmoid_probabilities > 0.5) # if probability is greate than '0.5' then return '1' else '0' hence we need
                                                                     # to manually convert them.



# Makeing Confusion matrix
# ------------------------
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix = confusion_matrix(y_test, y_predicted_binaryOutput)
accuracy = accuracy_score(y_test, y_predicted_binaryOutput)             # we have accuracy of 86%