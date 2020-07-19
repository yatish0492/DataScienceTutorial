'''
How Does 'Random Forest Classification' work?
    It is an enahancement of 'Decision Tree Classification'.
    This will  not take complete test set and train the ML. each time it will
    take a random set of inputs in the training set and creates a decision
    tree. Then it will pick a best decision tree among those and use that.
    
NOTE: the program is same as '3_1_SVM' even the data is same. we are just using
      'RandomForestClassifier' instead of 'svm' that's all
      
NOTE: unlike svm it can have seperate circles or shapes to classify the points.
      to know what this means execute this program and see the graph.
    
'''
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# DataSet description
# -------------------
# Data comprises users information like gender, age, salary and whether they purchased a product advertised 
# in online advertisement or not.

# Importing the dataset
dataset = pd.read_csv('/Users/ycs/PycharmProjects/DataScieneTutorial/Data/3_1_SVM.csv')
x = dataset.iloc[:,[2,3]].values # we are ignoring first column which is userId
y = dataset.iloc[:,4].values


# Spliting the dataset into training and test set
# -----------------------------------------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.8, random_state = 0)
# if we mention 'test_size = 0.2' then we get accurace of 68%. by giving 'test_size = 0.8', we got 87%. surprising!!!


# Feature Scaling
# ---------------
from sklearn.preprocessing import StandardScaler
standardScalarObj = StandardScaler()
x_train = standardScalarObj.fit_transform(x_train)
x_test = standardScalarObj.fit_transform(x_test)


# Fitting logistic regression to the training set
# -----------------------------------------------
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
                                    # n_estimators --> this means number of trees we want to create
                                    #                  basically it number of iterations we want to do.
                                    # criterion --> need to check what it is. 'entropy' normal english 
                                    #               meaning is 'lack of order' basically we are classifying
                                    #               the items which are not in order.
                                    
classifier.fit(x_train, y_train)



# Predicting test set result
# --------------------------
y_predicted = classifier.predict(x_test)


# Making Confusion Matrix
# -----------------------
from sklearn.metrics import confusion_matrix, accuracy_score
confusionMatrix = confusion_matrix(y_test, y_predicted) # remember 1st argument should be actual result and next should be prediction.
accuracy = accuracy_score(y_test, y_predicted) # accuracy_score is a tool to calculate accuracy.


# Visualizing the classification
# ------------------------------
# x-axis will represent 'age' and y-axis will represent 'salary' of the customer. each dot represents a customer.
# ML algorithm will draw a line seperating 2 areas. One area means the customer in that zone will buy the product and other area
#   means those customers won't buy the product.
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results
# --------------------------------
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()