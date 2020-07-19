'''

Convolutional ANN (CNN)
-----------------------
	Convolutional ANN is usually used for processing images. Images have huge number of pixels hence it will reduce
	the pixels by taking out un-necessary pixels required to classify the image.

    NOTE : Convolutional ANN is called as CNN
'''

# Importing libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# -----------------------------------------------------------------------------
#                       Training the training dataset
# -----------------------------------------------------------------------------

# Transforming the images
# -----------------------
# The 'ImageDataGenerator' is the class that is used to transform the data/image
# The parameter of the 'ImageDataGenerator' class are as follows,
#       rescale --> This is for feature scaling. here value used is '1./255'. 
#                   what this means is we will divide each pixel value by '255' 
#                   so that we have the normalized data for ML to process.
#       shear_range, zoom_renage, horizontal_flip--> this is used to avoid 
#                       over-fitting of images. basicallythese parameters 
#                       are used for reducing the pixels anda arriving at 
#                       'pooled Feature Map' images.
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True
                )


# Training dataset
# ----------------
# 'flow_from_directory' is the method used to supply the data directory and other parameters
#  The parameter of 'flow_from_directory' is explained as follows,
#           target_size  --> It is the actual size  of the image which will be before the
#                            image is fed to ANN. i.e. the image size after 'pooling' step
#           batch_size -->   Number of images taken together for each iteration of adjusting
#                            the weights. i.e. 'Stochastic gradient descent'
#           class_mode -->  In this case, the output is only 2 categories 'Dog' or 'Cat' hence,
#                           we are specifying 'binary'. if it is more than 2 categories then 
#                           we need to specify 'categorical' as value.
training_set = train_datagen.flow_from_directory(
                    '/Users/ycs/PycharmProjects/DataScieneTutorial/Data/8_3_ANN_Convolutional/training_set',
                    target_size=(64,64),
                    batch_size=32,
                    class_mode='binary'
                )


# -----------------------------------------------------------------------------
#                       Preparing the test dataset
# -----------------------------------------------------------------------------

# Transforming the images
# -----------------------
# we are not passing 'hear_range','zoom_range','horizontal_flip' here because 
# we are not training again on the 'test' data. So we will just only use 
# 'rescale'. It is like similar to 'fit_transform', we use 'fit_transform' only
# on 'training' set but on test set we use only 'transform'
test_datagen = ImageDataGenerator(
                    rescale=1./255
                )


# Test dataset
# ------------
# We will use same 'target_size' as that of used in training because ANN was 
# trained on the same format/pixels size of image so.
test_set = test_datagen.flow_from_directory(
                '/Users/ycs/PycharmProjects/DataScieneTutorial/Data/8_3_ANN_Convolutional/test_set',
                target_size=(64,64),
                batch_size=32,
                class_mode='binary'
    )



# -----------------------------------------------------------------------------
#                       Initializing CNN
# -----------------------------------------------------------------------------

# Creating a sequence for layers of CNN
# -------------------------------------
cnn = tf.keras.models.Sequential()


# Adding Convolution Layer
# ------------------------
# We us 'tf.keras.layers.Conv2D' class to add Convolution Layer. We need to pass 4 parameters
#       filters --> it is the number of 'Feature Detectors' we want to apply. here we are 
#                   giving '32' hence it uses 32 'Feature Detectors'
#       kernel_size --> the is the size of the 'Feature Detector' in this case we are 
#                       considering 3*3 matrix but while giving value we give only '3'
#       activation --> This is the activation function to use. in case of classification its
#                      not linear hence we use 'rectifier' function i.e. 'relu'
#       input_shape --> This is the image input details about the image which goes to CNN 
#                       after 'Pooling'. in this case, input value is [64, 64, 3]. 64,64 is
#                       the pixel of the images and '3' means its color image hence 3 colors 
#                       RGB. if it was black and white then it would be '1'       
#        
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64, 64, 3]))



# Pooling
# --------
# We are using 'Max Pooling' here, we use 'tf.keras.layers.MaxPool2D' class for this pooling layer.
# we pass following parameters to 'MaxPool2D' class,
#       pool_size --> This is the size of image we need to consider for pooling. here we have 
#                       specified 2. that means 2*2 matrix will be considered for max pooling.
#       strides --> This is the number of columns 2*2 matrix should move next to find max
#                   value. basically it is like sliding window value. here 2 means 2*2 matrix
#                   will move 2 columns after finding max element in current matrix.
#   
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2 ))


# Flattening
# ----------
# We are flattening the 'Max Pooled' images into flat list of numbers. which is required for 
# CNN input layer.
cnn.add(tf.keras.layers.Flatten())


# Full Connection - Adding Input layers of CNN
# --------------------------------------------
# # We use 'Dense' class object which creates a layer. 
# The parameters of 'Dense' class constructor explained below,
#   units --> Number of neurons. there is no rule of thumb to decide number of neurons. it is based on experimentation and 
#                 arrive at a number which gives more accuracy.
#   activation --> activation function. this uses 'rectifier' activation
#                   function. 'relu' is the code name for 'rectifier' activation function.
cnn.add(tf.keras.layers.Dense(units = 28, activation = 'relu'))


# Adding Output Layer of CNN
# --------------------------
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
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))



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
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# fitting ANN (Training)
# ----------------------
# One difference here is we are providing 'test' data as well along with training data.
#    'epochs'     --> Number of iterations of checking error percentage and re-calculating the wieights and checking error percentage.
#                        basically the iterations of 'Stochastic gradient descent' in this example.
#
# NOTE : we didn't give 'batch_size' here because, we have already given this while creating the 'training_set' above. 
#        
cnn.fit( x = training_set, validation_data = test_set, epochs = 25)


# Making single prediction
# ------------------------
#
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/Users/ycs/PycharmProjects/DataScieneTutorial/Data/8_3_ANN_Convolutional/test_set/dogs/dog.4001.jpg', target_size = (64, 64))
                                                        # We are getting the image which we want to test. we specify 'target_size' as (64,64) because, 
                                                        # we trainied the images with this resolution(64,64).
test_image = image.img_to_array(test_image)     # converting the image matrix to array of numbers. which is required for cnn network.
test_image = np.expand_dims(test_image, axis = 0)   # when we trained the data, we trained as batches of 32 images. Now we are trying to predict for 1 image
                                                    # but cnn is capable of predict in batches only as it was trained with batches hence we have to create a 
                                                    # batch with this 1 image. we do this by creating one more dimension using 'np.expand_dims' which means 
                                                    # expand dimension. we are giving 'axis=0' which means this axis has batch of images, say like if we give
                                                    # 'np.expand_dims(test_image, axis = 0)' then it creates batch with image 'test_image'. we can give following,
                                                    # 'np.expand_dims([test_image, test_image1, ...., test_imageN], axis = 0)' this will create batch of images 
                                                    # mentioned in the array. it will create matrix of one row each element in that row is corresponding image.
                                                    
result = cnn.predict(test_image)
test_image_result = result[0][0]        # Output --> 1. which means '1' is index for dog. 0 is index of 'Cat' if we have 2 images in batch then result[0][0] 
                                        #                contains result of 1st image and result[0][1] will have result of 2nd imge.
                                        # because we have only one image in axis=0 hence its 'y' index is also '0'. so we are getting result[0][0].























