'''

What is Natural Language Processing?
    Applying machine learning models to text and voice. Literally teaching machines to understand normally spoken and
    written words.
    eg: When you input voice to your phone it types it
        siri
        Analyzing a text book and classifying it like technical, naval or any other category.
        etc.

Which are the mail NLP libraries used?
    1) Natural Language Toolkit (NLTK)
    2) SpaCy
    3) Stanford NLP
    4) OpenNLP


What are the steps involved in the natural language processing?
    1) Text cleaning
            Text cleaning means say like we need take out 'The' or 'in' kind of words which are not useful in judging,
            whether the review is positive or not. We also make all text to lowercase instead of having 'Loved', 'love',
            'Love' etc.
    2) Steming
            Consider there are many words in the training set of reviews like 'loved', 'loving', 'love' etc. then
            instead of having all these words for predicting a review we can use only 'love'.
    3) Bag of words/texts
            Each word in each review will be taken and created a matrix of each column for each unique words in all
            reviews. The rows will point to the number of reviews. say like for first review it will mark row '0'
            with 1 for the corresponding keywords column which it has.


    * all example or explanation are given relative to considering that Natural language processing is trying to find
        whether if the review is good or bad.
'''