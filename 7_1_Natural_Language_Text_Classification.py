'''

Dataset --> The data have details about the customer review description and whether they liked it or not.

NOTE: Natural Language Processing falls under 'Classification' ML.

'''

import pandas as pd


# we are specifying 'delimiter' because we are reading 'tsv' file. usually 'csv' is called 'comma seperated values' but in this case,
# we are using 'tsv' file which is 'tab seperated values' so. 'read_csv()' method's default delimiter is 'comma' hence we need to explicitly
# specify that for this file delimiter is 'tab'.
# we are specifying 'quoting' value to 3 which means it tells 'read_csv()' method to ignore 'double quotes'.
dataset = pd.read_csv('/Users/ycs/PycharmProjects/DataScieneTutorial/Data/7_1_Natural_Language_Text_Classification.tsv', delimiter = '\t', quoting = 3)


# Cleaning the text
# ------------------
# 're' is the library that helps us clean the text.
import re
oneReviewBeforeCleaning = dataset['Review'][0] # 'Wow... Loved this place.'
oneReviewAfterCleaning = re.sub('[^a-zA-Z]', ' ', oneReviewBeforeCleaning) # 'Wow    Loved this place'
                        # basically replaces '[^a-zA-Z]' with space ' '. that means apart from any characters from
                        # characters will be replaced by space.
oneReviewAfterCleaningLowerCase = oneReviewAfterCleaning.lower() # making all words to lowercase.


import nltk
nltk.download('stopwords')  # This will download those words when executed as they are not part of standard package.
                            # 'stopwords' have all the words which are useless for predicting the nature of the sentence,
                            # like 'this', 'that', 'in' etc. these are useless for predicting whether if the review is 
                            # positive or negative.
from nltk.corpus import stopwords # in previous step we downloaded it and here we are importing it.
 
reviewWordsList = oneReviewAfterCleaningLowerCase.split()

reviewWordsListRemovedUnnecessaryWords = [word for word in reviewWordsList if not word in stopwords.words('english')]
                        # reviewWordsListRemovedUnnecessaryWords = ['wow', 'loved', 'place']. 
                        # 'this' is removed from the list of words.
                                # this is like lambda 'filter' expression of java. we are iterating over all the words in 'reviewWordsList'
                                # and then filtering out the words not in 'stopwords'
                                # we are specifying 'english' because 'stopwords' have all locale words so.
                                # we can put 'stopwords.words('english')' inside 'set()' like 'set(stopwords.words('english'))'
                                # as the 'list' will be slower compared to 'set'


# stemming
# ---------
# 'PorterStemmer' is a class which does stemming for us.
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
stemmedReviewWordsList = [ps.stem(word) for word in reviewWordsListRemovedUnnecessaryWords]
                        # stemmedReviewWordsList = ['wow', 'love', 'place']
                        # 'ps.stem()' has changed 'loved' to 'love'.
                        

joinedReviewWords = ' '.join(stemmedReviewWordsList)



#******************************************************************************************************************
# Above was to show only for one row for understanding purpose. Now lets do it for all the rows.
#******************************************************************************************************************

import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
dataset = pd.read_csv('/Users/ycs/PycharmProjects/DataScieneTutorial/Data/7_1_Natural_Language_Text_Classification.tsv', delimiter = '\t', quoting = 3)
corpus = []     # In natural langugae ML, we usually name the variable 'corpus' standard practice which is collection of texts.
ps = PorterStemmer()
for i in range(0, 1000):
    eachText = dataset['Review'][i]
    eachText = re.sub('[^a-zA-Z]', ' ', eachText)    # 're' is a replace function. it will replace any character other then alphabets with ' '
    eachText = eachText.lower()
    eachText = eachText.split()
    eachText = [word for word in eachText if not word in set(stopwords.words('english'))] # removes all the stop words like 'this' etc.
    eachText = [ps.stem(word) for word in eachText]
    eachText = ' '.join(eachText)
    corpus.append(eachText)
    
# Create Bag of words model
# -------------------------
# Consider we are analysing 3 reviews to train ML
#       1st review --> 'wow love place'     -> positive
#       2nd review --> 'crust good love'    -> positive  
#       3rd review --> 'tasti nasti  -> negative
# Now creating bag of words model will create a sparse matrix for this as follows,
#
#           wow   love   place   crust   good   tasti  nasti   Final_Result 
#       0    1      1     1       0        0     0       0        1
#       1    0      1     0       1        1     0       0        1
#       2    0      0     0       0        0     1       1        0
#
#  as we can see for 'love' word '1' is specified in both 1st and 2nd row as it is present in 1st and 2nd review. Basically we arrive at a standard
# classification form, which have only numbers(matrix) which can be understandable by ML algorithms to apply its mathematical operations. 
# whereas ML cannot understand apply its mathematical operations on text/words.

# We use following library/tool to create the 'Bag of words'/'sparse matrix'
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)  # 'max_features' is the property which specifies how many columns to be considered. i.e number of words.
                                           # we have '1565' words if we specify '1500' then '65' words will not be considered for fitting the ML.
                                           # It will automatically remove '65' words whose frequencies are less compared to other words. For deciding
                                           # what this number should be, it depends on user. Say like i want to exclude words whose freqency is less
                                           # than '10'. then we will count how many words which have frequency less than '10' and put reduce that count
                                           # from existing words and put that number as value.
# we manually did the replace of any other characters other than alphabets to empty space and then we changed them to lowercase and then we removed 
# the stop words manually. we also could have done that by passing following parameters to 'CountVectorizer' as shown below,
#    cv = CountVectorizer(stop_words={‘english’}, lowercase=True, token_pattern='[^a-zA-Z]')
#           stop_words={‘english’} --> This will remove all the stopwords while creating bag of words
#           lowercase=True --> This will transform all of them to lowercase
#           token_pattern='[^a-zA-Z]' --> this will make sure that any words which doesn't match this regex are removed.
sparseMatrix = cv.fit_transform(corpus).toarray() # this will give us the 'Bag of words'/'sparse matrix'
CommentResult = dataset.iloc[:,1].values  # this column specifies result. i.e. whether positive or negative comment.


# Training the ML
# ---------------
# This is a classification ML. as we need to clasify whether if comments is of positive or negative.
# Based on experimentation of all classification MLs, scientists have suggested that 'Naive Bayes', 'Decision Tree Classification' and
# 'Random Forest Classification' are best for Natural language processing as these 3 give us more accurate results so.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sparseMatrix, CommentResult, test_size = 0.2, random_state = 0)



# Using random forest classification
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

    
    

    
