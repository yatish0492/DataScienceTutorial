'''
What is Classification?
-----------------------
Unlike regression where you predict a continuous number, you use classification to predict a category. There is a wide variety of classification 
    applications from medicine to marketing. Classification models include linear models like Logistic Regression, SVM, and nonlinear ones like K-NN, 
    Kernel SVM and Random Forests.
    
eg: we have data of customers(age, salary etc) who bought a fridge and did not from a store. then we can predict whether a new customer will buy it or not.
    basically we are classifying the customers into 2 catagories. we are not predicting a coutinuous number like regression. hence it is called classification.  
    

Why do we call few classification models are linear and non-linear?
-------------------------------------------------------------------
When we visualize the ML, it actually draws a line seperating categories. For linear models, this line will be straingt/linear. 
    For non-linear models, this will will not be straight/linear it can be curvy or some other shape etc.
    

Whether if its linear or non-linear model how does it matter?
-------------------------------------------------------------
Consider in linear model, it draws a line seperating 2 categories and there is one point which doesn't belong to that category,
    in that case, if we use non-linear model then, we can go around it so that it belongs to correct category. but in case of linear
    model we cannot make curve so that point will be left in wrong side of the category.
           
           Linear Model
           ------------
eg:     |        \                                     
        |         \   orange
        |   apple  \              orange
        |           \      
        |            \         orange        --> Linear model, we know that
        |      orange \                          one 'orange' is in 'apple'
        |              \                         category but still because
        |               \      orange            we cannot have bent lines
        |   apple        \                       in linear model we can't do
        |                 \                      anything.
        |                  \
        --------------------------------
            
            Non-Linear Model
            ----------------
        |        \                                     
        |         \   orange
        |   apple  \              orange
        |           \      
        |    --------          orange        --> Non-Linear model, we bent 
        |    | orange                            the line so that 'orange'
        |    -----------                         can belong to its correct
        |               \      orange            category.
        |   apple        \                       
        |                 \                      
        |                  \
        --------------------------------
        
        
So always non-linear models are better right as they give more accurate result. then why do we need/have linear models?
-----------------------------------------------------------------------------------------------------------------------
Don't know ********* Need to explore ***********


What is Confusion Matrix?
-------------------------
After we prepare an ML which predicts the result. how do we validate whether ML is predicting correctly or not. We cannot keep validating each predicted result
    with actual result as data will be huge so 'Confusion Matrix' is a library of 'sklearn' which will help us validate this. 
    
    Confusion matrix gives a matrix of 2*2. in which left top to right botom diagonal values show correct predections and right top to left bottom diagonal
    values show incorrect predictions.
    
eg: | 10  2  |   --> This matrix means we have (10+50 = 60) correct predictions and (2+3 = 5) in-correct predictions.
    | 3   50 |
    
    consider customer buying fridge example, 
        10 -> this value tells like for 10 cases, customer bought fridge and ML predicted correctly that he bought the fridge.
        2  -> this value tells like for 2 cases, customer bought fridge but ML predicted in-correctly that he didn't bought the fridge.
        3  -> this value tells like for 3 cases, customer didn't bought the fridge but ML predicted in-correctly that he bought the fridge.
        50 -> this value tells like for 50 cases, customer didn't bought the fridge and ML predicted correctly that he didn't bought the fridge.
    1st row will be for actual 'positive' case, that means for the cases in which customer bought fridge.
    2nd row will be for actual 'negative' case, that means for the cases in which customer didn't bought the fridge.
    
    in this example accuracey is 60/(60+5) = 0.92 --> 92%
'''