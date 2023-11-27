#! /usr/bin/env python

import pandas as pd
import numpy as np
import re


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
import seaborn as sn


def label_data():
    rows = pd.read_csv('dataset.csv', header=0, index_col=False, delimiter=',')
    labels = []
    for cell in rows['Rating']:
        if cell >= 4:
            labels.append('2')   #Good
        elif cell == 3:
            labels.append('1')   #Neutral
        else:
            labels.append('0')   #Poor
            
    rows['Label'] = labels
    return rows

def clean_data(data):
    #columnwise print number of rows containing blank values
    #print data.isnull().sum()
    
    #replace blank values in all the cells with 'nan'
    data.replace('',np.nan,inplace=True)
    #delete all the rows which contain at least one cell with nan value
    data.dropna(axis=0, how='any', inplace=True)
    
    #Check the number of rows containing blank values. This should be zero now as compared to first line of this function
    #print data.isnull().sum()
    #save output csv file
    data.to_csv('labelled_dataset.csv', index=False)
    return data

def cleanText(text, remove_stopwords=False, stemming=False, split_text=False):
  
    letters_only = re.sub("[^a-zA-Z]", " ", text)  # remove non-character
    words = letters_only.lower().split() # convert to lower case 
    
    if remove_stopwords: # remove stopword
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        
    if stemming==True: # stemming
        stemmer = SnowballStemmer('english') 
        words = [stemmer.stem(w) for w in words]
        
    if split_text==True:  # split text
        return (words)    
    return( " ".join(words))

def modelEvaluation(predictions, y_test_set):
    #Print model evaluation to predicted result 
    
    print("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test_set, predictions)))
    print("\nClassification report : \n", metrics.classification_report(y_test_set, predictions))
    print("\nConfusion Matrix : \n", metrics.confusion_matrix(y_test_set, predictions))


data = label_data()
data = clean_data(data)
print(data.head())



#split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(data['Reviews'], data['Label'], test_size=0.1, random_state=0)
# Preprocess text data in training set and validation set
x_train_cleaned = []
x_test_cleaned = []
y_test_cleaned = []

for d in x_train:
    x_train_cleaned.append(cleanText(d))

for d in x_test:
    x_test_cleaned.append(cleanText(d))  

for d in y_test:
    y_test_cleaned.append(cleanText(d))  

print(x_test_cleaned)
print(y_test_cleaned)


# Fit and transform the training data to a document-term matrix using TfidfVectorizer 
tfidf = TfidfVectorizer(min_df=5) #minimum document frequency of 5
x_train_tfidf = tfidf.fit_transform(x_train)


# Logistic Regression
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(x_train_tfidf, y_train)
lr_predicted = lr.predict(tfidf.transform(x_test_cleaned))
print(lr_predicted)


modelEvaluation(lr_predicted, y_test)
cm=metrics.confusion_matrix(y_test, lr_predicted)
df_cm = pd.DataFrame(cm, range(3), range(3))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.show()

# RandomForest Classifier
rand = RandomForestClassifier()
x_train_input = tfidf.transform(x_train_cleaned)
rand.fit(x_train_input, y_train)
rand_predicted = rand.predict(tfidf.transform(x_test_cleaned))
print(rand_predicted)
modelEvaluation(rand_predicted, y_test)
cm=metrics.confusion_matrix(y_test, rand_predicted)
df_cm = pd.DataFrame(cm, range(3), range(3))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.show()
