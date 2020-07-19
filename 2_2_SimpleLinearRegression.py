import pandas as pd
import matplotlib.pyplot as plt


# Importing dataset
# -----------------
# Dataset has Years of Experience and Salary column of employees
dataset = pd.read_csv('/Users/ycs/PycharmProjects/DataScieneTutorial/Data/2_2_LinearRegression_Example.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


# Splitting the data into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)


# Feature Scaling
# ---------------
# Why are we not doing Feature Scaling here? 
#   Actually the ml model 'LinerRegression' used below will automatically do that.


# Fitting Simple Linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# Predicting the test set results
y_predicted = regressor.predict(x_test)


# Visualizing the Training Set results
plt.scatter(x_train, y_train, color='red') # 'scatter' means it will show dots in the graph
plt.plot(x_train, regressor.predict(x_train), color='blue')  # 'plot' means it will connects the dots and draw the line.
                                                             # This is the linear line which is predicted by linear regression model,
                                                             # based on the training done in 'regressor.fit(x_train, y_train)'
plt.title('years of Experience vs Salary(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualizing the Test Set Results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue') # linear line predicted for 'training' data
plt.plot(x_test, y_predicted, color='yellow')               # linear line predicted for 'test' data
plt.title('years of Experience vs Salary(Test Set)')        # Result --> Both the lines are same. overlapping with each other.     
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

