import pandas as pd
import numpy as np
import re


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,precision_recall_curve,PrecisionRecallDisplay,RocCurveDisplay,f1_score,roc_curve,accuracy_score
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt



import mysql.connector
import csv
mydb = mysql.connector.connect(host="localhost",user="root",password="",database="shopping")

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM productreviews")

rows = mycursor.fetchall()
column_names = [i[0] for i in mycursor.description]
fp = open('train.csv', 'w')
myFile = csv.writer(fp, lineterminator = '\n')
myFile.writerow(column_names)   
myFile.writerows(rows)
fp.close()

def clean_data(data):
    data.replace('',np.nan,inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    data.to_csv('train.csv', index=False)
    return data





data = pd.read_csv('train.csv', header=0, index_col=False, delimiter=',')
data = clean_data(data)
print(data.head())



#split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(data['review'], data['productId'], test_size=0.2, random_state=0)


# Fit and transform the training data to a document-term matrix using TfidfVectorizer 
tfidf = TfidfVectorizer() #minimum document frequency of 5
x_train_tfidf = tfidf.fit_transform(x_train)



#Algorithm comparison
algorithms = {"SVM Classifier:":SVC(kernel='linear'),"RandomForestClassifier":RandomForestClassifier(),"KNeighborsClassifier":KNeighborsClassifier()}

results = {}
for algo in algorithms:
    clf = algorithms[algo]
    x_train_input = tfidf.transform(x_train)
    clf.fit(x_train_input, y_train)
    ypredt=clf.predict(tfidf.transform(x_test))
    score = accuracy_score(y_test,ypredt,normalize=False)
    score=score/25
    print("%s : %f %%" % (algo, score*100))
    results[algo] = score
print(results)


bestalgo=max(results,key=results.get)
print(bestalgo)
classifier=algorithms[bestalgo]
x_train_input = tfidf.transform(x_train)
classifier.fit(x_train_input, y_train)
classifier_predicted = classifier.predict(tfidf.transform(x_test))
print(classifier_predicted)

output_predicted=classifier_predicted.tolist()
print(output_predicted)
##for bst in output_predicted:
##    #print(bst)
##    mycursor = mydb.cursor()
##    sql = "INSERT INTO bestproduct (productid) VALUES (%s)"
##    val = [(str(bst))]
##    mycursor.execute(sql, val)
##    mydb.commit()
##
##
##
##
############################   GRAPH    ################################
##
##print("cmplot")
##cm = confusion_matrix(y_test, classifier_predicted, labels=classifier.classes_)
##disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
##disp.plot()
##plt.title("CONFUSION MATRIX")
##plt.show()
