'''
Linear Regression
-----------------
Linear Regression is a machine learning algorithm based on supervised learning.


What are the types of Linear Regression?
----------------------------------------
1) Simple Linear Regression
2) Multiple Linear Regression


Simple Linear Regression
------------------------
Consider, an example, we have data about the salary of the employees currently working in the company along with their experience.
    Let us plot a chart for this data with x-axis as the experience and the y-axis as the salary. The Simple Linear Regression algorithm will
    find a line using the formula,
            'y = b0 + b1*x'
                b0 --> This is the base value say like that is the minimum salary that is given to a fresher/entry-level employee so any 
                        salary will be always equal to or greater than this salary only
                b1 --> This is a co-efficient value. This value is the slope of the line that will be formed. This value is found by using
                        some mathematical formulas using sum of standard deviation of the values and dividing them with somthing and all. Basically
                        we take the available salary data and apply the formula on that and arrive at this value.
                x --> This is the salary of the employee.
                
    In the chart, If we draw lines between all the 'x' values connecting them then we get a zig zag line, using which we cannot do salary predictions. 
        So we calculate 'y' at each point 'x' and put the 'y' points on the chart. Then we connect all the 'y' points which will form a line,
        which will be straight line. So using this straing line we can extend it further and predict the salary based on any experience.
        
    In general, the straing line formed by connecting 'y' values will be like a 'mean' line which makes it straingt instead of zig zag by taking
        'mean' values of x. It is not exactly 'mean' values of 'x' it will be based on the formula 'y = b0 + b1*x'
        
    By varying the co-efficient value 'b1' we can end up in different straint lines, the model will select the line which have less distance between
        'x' and 'y' value in the chart for all the 'x' points(salary). Basically, that means it will take line which gives the predicted salaries(y)
        almost same as the real salary 'x', basically it will consider a line which gives more accurate prediction.
        

Multiple Linear Regression
--------------------------
It is similar to 'Simple Linear Regression' but in this, instead of only one indipendent variable like 'experience', if the usecase have mutliple independent 
variables like 'technology', 'designation' along with 'experience'. Then we use Multiple Linear Regression.

The Formula will be something like,
             'y = b0 + b1*x1 + b2*x2 + b3*x3 ...'
                 b0 --> base value
                 b1 --> Co-efficient value of x1
                 x1 --> 'technology'
                 b2 --> Co-efficient value of x2
                 x2 --> 'designation'
                 b3 --> Co-efficient value of x3
                 x3 --> 'experience'
    
'''