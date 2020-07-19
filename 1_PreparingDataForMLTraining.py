'''
Importing the libraries
'''
import numpy as np                 #used for mathematical functions 
import matplotlib.pyplot as plt    #used to plot nice charts
import pandas as pd                #used to import datasets and manage them

'''
Importing the dataset
'''
dataset = pd.read_csv('/Users/ycs/PycharmProjects/DataScieneTutorial/Data/1_PreparingDataForMLTrainingData.csv')
# indexes withing iloc[] are coma seperated. before comma specifies rows and after comma specifies column.
x = dataset.iloc[:, :-1].values       #Imports all columns except last column. i.e. column with index '-1'
y = dataset.iloc[:, 3].values         #Imports only 3rd Column. starting column with index 0.
z = dataset.iloc[:-2,:].values        #Imports all rows except last 2 rows
# NOTE: 'iloc[...].values' returns type 'ndarray'. if we use only 'iloc[...]' then it returns DataFrame Type.

'''
Filling Missing data
---------------------
Usually when there is a value missing in the dataset then python will automatically put 'nan' there.
'''
# Scikit-learn is a library that provides many unsupervised and supervised learning algorithms
# 'SimpleImputer' is a class
from sklearn.impute import SimpleImputer 
# missing_values --> specify what value in data which specify it is empty/missing. (default = mean)
# strategy --> specifies what logic should be used to fill the missing values. 'mean/median/most_frequent/constant' of other values of column.
imputerObj = SimpleImputer(missing_values = np.nan, strategy="mean")
imputerObj = imputerObj.fit(x[:, 1:])
x[:,1:] = imputerObj.transform(x[:, 1:]) # we could have directly used imputerObj.fit_transform(x[:, 1:]) instead of 2 steps


'''
Encoding categorical data
--------------------------
Consider we have a column of countries(strings) and you want to include them in calculations of ml then we cannot process strings in our
mathematical calculations so we need them as numbers. Encoding these to numerical value is called Encoding categorical data.
There are 2 encoders as follows,
1) LabelEncoder
2) OneHotEncoder
NOTE: we are able to send strings also to 'train_test_split()' and getting result but that might not be accurate so better we encode.
'''
'''
LabelEncoder
------------
Label encoder is more suited when we are encoding columns with only 2 different values like 'Gender' which has Male/Female
'''
from sklearn.preprocessing import LabelEncoder
labelEncoderObj = LabelEncoder()        # No 'new' keyword required in python.
# LabelEncoder will take the contents of the column in this case, unique countries in the column are 'France','Spain','Germany'
# so LabelEncoder will encode them by assigning a number/index to them as follows,
# 'France' --> 0
# 'Spain' --> 2
# 'Germany' --> 1
# it will replace the strings with number/index/code 0,1,2
# it basically returns a list of numbers based on encoding. we can assign it to 'x' dataset to replace strings with code as follows,
# x[:,0] = labelEncoderObj.fit_transform(x[:,0]);
# NOTE: we should encode 'Purchased' column as well so that we have 1 or 0 instead of 'yes' or 'no'
# NOTE: even though we can encode 'Countries' using label encoder, if the values contain more than 2 categories(countries).
countrylabelEncoded = labelEncoderObj.fit_transform(x[:,0]);   # Returns an array
'''
OneHotEncoder
-------------
 We had labelEncoder right, then why do we need this oneHotEncoder?
     We should use labelEncoder only if the values in the column are of only 2 categoires like 'Yes' or 'No' etc.
 If the values of column contains more than 2 categories like 'France','Spain','Germany', then we have to use oneHotEncoder.
 
 What happens if we use labelEncoder instead of oneHotEncoder for column containing more than 2 categories?
     The Machine learning algorithms sometimes mis-interpretes the values like for example, If 'Country' column contains
 0,1,2 then it might consider 2 is the biggest numbers so it is more preferred and might start returning 2 --> 'Spain' only
 for all the inputs it receive after, which is incorrect. We have just encoded 2 for 'Spain' to just identify it with that index.
'''
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
# 'OneHotEncoder' is similar to 'labelEncoder' but it will return 3 columns 0,1,2. columns representing as follows,
 # 'France' --> 0
# 'Spain' --> 2
# 'Germany' --> 1
# Thern each row will have 1 for column 0 if it is 'France' and 0 value for column 1,2.
# The output can be see in value of 'countryLabelEncodedIntoMultipleColumns'
columnTransformer = make_column_transformer(
    ( OneHotEncoder(), ['Country'])          # input is a tuple of transformer and columns.
)
countryLabelEncodedIntoMultipleColumnsDroppingOtherColumns = columnTransformer.fit_transform(dataset)
# 'make_column_transformer' have one more parameter called 'remainder' this accepts 2 values,
#   remainder = "drop" --> this is the default value. this will drop other columns apart from the columns mentioned in 'make_column_transformer'
#   remainder = "passthrough" --> this will not drop other columns and keep them as it is in the result of 'fit_transform()' and it will just 
#                                  replace the columns mentioned in 'make_column_transformer' with corresponding encoded columns.
# eg: if you want to see the difference then execute this program and see values of 'countryLabelEncodedIntoMultipleColumnsDroppingOtherColumns'
#       and 'countryLabelEncodedIntoMultipleColumnsWithoutDroppingOtherColumns'
columnTransformer = make_column_transformer(
    ( OneHotEncoder(), ['Country']) ,         # input is a tuple of transformer and columns.
    remainder = "passthrough"
)
countryLabelEncodedIntoMultipleColumnsWithoutDroppingOtherColumns = columnTransformer.fit_transform(dataset)
# NOTE: we can either mention the column name or index number of it. in this case instead of 'Country' you could have mentioned [0]

# Another way of creating 'columnTransformer' instead of using 'make_column_transformer'
# --------------------------------------------------------------------------------------
# Actually 'make_column_transformer' is a shorthand/shortcut of creating 'ColumnTransformer'. it also calls 'ColumnTransformer' constructor 
#  internally. it will give names for transformers automatically when passing it to constructor. in below code we are manually passing it as
#   'countryEncoder'
#
# NOTE : Disadvantage of 'make_column_transformer' is it doesn't support 'transformer_weights' which we can pass in 'ColumnTransformer' constructor.
# NOTE : 'transformer_weights' --> Multiplicative weights for features per transformer. The output of the transformer is multiplied by these weights/values.
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer(transformers = [('countryEncoder', OneHotEncoder(), ['Country'])], remainder = "passthrough")
countryLabelEncodedIntoMultipleColumnsWithoutDroppingOtherColumns = columnTransformer.fit_transform(dataset)


'''
Splitting the dataset into training set and test set
----------------------------------------------------
Syntax --> train_test_split(<InputData(DataFrame)>, <OutputData(DataFrame/Series)>, testsize=<testsize/Ratio/percentage>, random_state=<Random_Value>)

Let us explain each parameter with help of example, Say there is table of columns 'Contry','Age','Salary' and 'Purchased'. in this case we want to 
train ML to predict that whether a person will purchage a commodity or not based on his 'Country','Age' and 'Salary'.

    <InputData(DataFrame)> --> we give only columns 'Country','Age' and 'Salary' in form of DataFrame
    <OutputData(DataFrame/Series) --> we give column 'Purchased' in form of DataFrame or Series
    testsize --> This is the percentage of data want to use for testing. say 0.2 means use 80% of data for training and test remaining 20%
                 NOTE: we should always give atleast 60% as train data. even 50% is also not good as there is possibility of more inaccuracy. 
                       with more data, we train ML more so ML learns more and more accurate it will be.
    random_state --> Usually, a random number generator is used with 'train_test_split' to take the samples for training and testing. each and
                     every time, if new random number is generated, then in each run it picks different random samples for training and testing.
                     eg: Consider if 1st 8 rows are taken for train data and last 2 rows are picked for test data. in next run it might take 
                         3rd and 4th rows for test data and remaining for train data.
                     'random_state' value will be directly used by random number generator so each and every time it generates same numbers so that
                     for each run of 'train_test_split', the train data and test data is same.
    
Return Value --> xTrain, xTest, yTrain, yTest
    
    xTrain --> Values of 'Country','Age' and 'Salary' that can be used for Training ML
    yTrain --> Corresponding 'Purchased' column values of xTrain. 
    xTest --> Values of 'Country','Age' and 'Salary' used to test ML after training.
    yTest --> Corresponding 'Purchased' column values of xTest.
                
When should we consider giving 'random_state' and when to not specify it?
???????????????? Need to explore it. as per the current understanding, sometimes the accuracy of test output or perforamnce will vary with 
different values so we might need to find a optimal value. But some places they have mentioned that it should be specified for only our
testing and not in production as sometimes the data changes and our value of new data might be less accurate so to not specify it in
production. ???????????????????????????

What is the need of doing 'train_test_split'?
    This is used to split the data into the training sets and tests sets so that we can use these sets for any ML training and test.
    
NOTE : don't think that 'train_test_split' does a ML and predicts 'yTest'. it is just the values from the input dataset. it will only
        just split the dataset into training set and test set based on the 'test_size' and other parameter we give.
                
'''
from sklearn.model_selection import train_test_split
InputDataFrame = pd.DataFrame(countryLabelEncodedIntoMultipleColumnsDroppingOtherColumns)

InputDataFrame['Age'] = x[:,1]
InputDataFrame['Salary'] = x[:,2]

OutPutSeries = pd.Series(y)

# Following are some of the useful methods of panda library. see examples below and explore
# -----------------------------------------------------------------------------------------
# InputDataFrame = InputDataFrame.append(x[:,1:3].tolist())
# InputDataFrame.append(dataset[['Salary','Age']], ignore_index = True)
# InputDataFrame = [InputDataFrame, dataset[['Salary','Age']]]
# pd.concat(InputDataFrame, ignore_index=True)
# InputDataFrame += x[:,1:3].tolist()

# We are able to send 'Country' as strings only instead of encoding them to integers using 'OneHotEncoder' to 'train_test_split()', we shouldn't
# even though we can as the incorrect training might happen due to string usage in mathematical functions of argorithms. 
#InputDataFrame = dataset.iloc[:,0:3]    # Returns DataFrame type
#OutPutSeries = dataset.iloc[:,3]

xTrain, xTest, yTrain, yTest = train_test_split(InputDataFrame, OutPutSeries, test_size = 0.2, random_state = 0)


'''
Feature Scaling
---------------
 In above example, 'dataset' have 'Age' and 'Salary' column.
 The 'Age' column have values with min 27 and max 50. The Diff between x2-x1 = 50 - 27 = 23
 The 'Salary' column have values with min 48000 and max 83000. The Diff between y2-y1 = 83000 - 48000 = 35000
 
 Since the difference of 'Salary' is 35000 and 'Age' is only 23. The huge difference of 'Salary' dominates over 'Age'.
 So 'Age' will not be considered in training of data, only 'Salary' is considered which is wrong, we want the training of model
 to be done considering both 'Age' and 'Salary' with equal importance. So we have to make sure the difference of max to min in all 
 the columns should be almost in same range so that they take equal importance in training the model.
 
 We use 'fit_transform', what does that mean?
    For 'Feature Scaling', we need few parameters like 'mean' and 'standard deviation' to be calculated. 
        fit() --> this will calculate 'mean' and 'standard deviation' for the feature/column and save them as internal object. 
        transform() --> this will use the 'mean' and 'standard deviation' calculated and stored in internal object bu 'fit()' and then
                        perform feature scaling on feature/column.
        fit_transform() --> this will internally call 'fit()' and then 'transform()'. this is just a shortcut rather than calling 'fit()'
                            and 'transform()' in 2 steps.
  NOTE: If you call 'transform()' without calling 'fit()' on those feature/column then it will give error as 'mean' and 'standard deviation'
          is not calcualted and stored in internal object, as 'transform()' needs them to calculate and do feature scalaing.

'''
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

preprocess = make_column_transformer(
    (['Age','Salary'], StandardScaler())          # input is a tuple of columns and transformer. The main idea is to 
                                                  # normalize/standardize (mean = 0 and standard deviation = 1) your features 
                                                  # before applying machine learning techniques. This will make sure that the
                                                  # columns 'Age' and 'Salary' Difference of Max and min are almost same.
)
AgeAndSalaryScalarized = preprocess.fit_transform(InputDataFrame)

xTrain, xTest, yTrain, yTest = train_test_split(InputDataFrame, OutPutSeries, test_size = 0.2, random_state = 0)



'''
What is formula of standardisation and normalisation?

stndardisation = (x - mean(x)) / Standard Deviation of (x)

Normalisation = (x - min(x)) / (max(x) - min(x))
'''












