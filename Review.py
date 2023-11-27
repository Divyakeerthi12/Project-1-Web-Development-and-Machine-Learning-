import pandas as pd
import numpy as np
import nltk
#import nltk
nltk.download('wordnet')

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
########################################################################################NLP################################################################################################
df=data
print(df.head())
df=df[['review','value','productId']]

df=df.dropna()
df = df.reset_index(drop=True)
print(df.head())


df['value']=df['value'].astype(int) #convert the star_rating column to int
df=df[df['value']!=3]
df['label']=np.where(df['value']>=4,1,0) #1-Positve,0-Negative

print(df.head())

print(df['value'].value_counts())  #Will count the unique values



import pandas as pd

df = df.sample(frac=1).reset_index(drop=True)  # shuffle

data_label_0 = df[df['label'] == 0].iloc[:400]
data_label_1 = df[df['label'] == 1].iloc[:400]

data = pd.concat([data_label_0, data_label_1], ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)  # shuffle again

print(data['label'].value_counts())
data


print(data)

#######################################PREPROCESSING##############################


data['pre_process'] = data['review'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))

from bs4 import BeautifulSoup
data['pre_process']=data['pre_process'].apply(lambda x: BeautifulSoup(x).get_text())
import re
data['pre_process']=data['pre_process'].apply(lambda x: re.sub(r"http\S+", "", x))

def contractions(s):
 s = re.sub(r"won’t", "will not",s)
 s = re.sub(r"would’t", "would not",s)
 s = re.sub(r"could’t", "could not",s)
 s = re.sub(r"\’d”", " would",s)
 s = re.sub(r"can\’t", "can not",s)
 s = re.sub(r"n\’t", "not", s)
 s= re.sub(r"\’re", " are", s)
 s = re.sub(r"\’s", "is", s)
 s = re.sub(r"\’ll", " will", s)
 s = re.sub(r"\’t", " not", s)
 s = re.sub(r"\’ve", " have", s)
 s = re.sub(r"\’m", "am", s)
 return s
data['pre_process']=data['pre_process'].apply(lambda x:contractions(x))
data['pre_process']=data['pre_process'].apply(lambda x: " ".join([re.sub('[^A-Za-z]+','', x) for x in nltk.word_tokenize(x)]))
data['pre_process']=data['pre_process'].apply(lambda x: re.sub(' +', ' ', x))

from nltk.corpus import stopwords
stop = stopwords.words('english')
data['pre_process']=data['pre_process'].apply(lambda x: " ".join([x for x in x.split() if x not in stop]))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
data['pre_process']=data['pre_process'].apply(lambda x: " ".join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)]))

print(data)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train, Y_test = train_test_split(data['pre_process'], data['label'], test_size=0.25, random_state=30)
print("Train: ",X_train.shape,Y_train.shape,"Test: ",(X_test.shape,Y_test.shape))


print("TFIDF Vectorizer……")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer= TfidfVectorizer()
tf_x_train = vectorizer.fit_transform(X_train)
tf_x_test = vectorizer.transform(X_test)
###################################################################################Support Vector Machine#######################################################################################################

from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=0)
clf.fit(tf_x_train,Y_train)
##print(tf_x_test)
y_test_pred=clf.predict(tf_x_test)

#print(y_test_pred)



from sklearn.metrics import classification_report
report=classification_report(Y_test, y_test_pred,output_dict=True)
print(report)

#####################################################################ACCURACY EXTRACTION###################################################################################################
from sklearn.metrics import classification_report



# Function to parse the classification report dictionary and extract accuracy
def extract_accuracy(classification_report_dict):
    accuracy = classification_report_dict['accuracy']
    return accuracy

# Convert the report dictionary to a string
report_str = classification_report(Y_test, y_test_pred,output_dict=True)

# Extract accuracy from the report
accuracy = extract_accuracy(report)
print("Accuracy:", accuracy)










##################################################################################LOGISTIC REGRESSION###########################################################################################################

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000,solver='saga')

clf.fit(tf_x_train,Y_train)
y_test_pred=clf.predict(tf_x_test)
#print(y_test_pred)

from sklearn.metrics import classification_report
report1=classification_report(Y_test, y_test_pred,output_dict=True)
print(report1)
###################################################################################ACCURACY EXTRACTION################################################################################
from sklearn.metrics import classification_report



# Function to parse the classification report dictionary and extract accuracy
def extract_accuracy(classification_report_dict):
    accuracy = classification_report_dict['accuracy']
    return accuracy

# Convert the report dictionary to a string
report_str = classification_report(Y_test, y_test_pred,output_dict=True)

# Extract accuracy from the report
accuracy1 = extract_accuracy(report1)
print("Accuracy:", accuracy1)
##################################################################################ALGORITHM GRAPH#######################################################################################

algorithm_names = ['SVM', 'Logistic Regression']  # Replace with your algorithm names
accuracies = [accuracy,accuracy1]  # Replace with your algorithm accuracies
import matplotlib.pyplot as plt
plt.bar(algorithm_names, accuracies)
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Accuracy of Algorithms')
plt.show()



######################################################################################DATABASE PART######################################################################################
new_data=data[['productId','label']]
#print(new_data)
import pandas as pd


df = pd.DataFrame(new_data)
filtered_df = df[df['label'] == 1]
counts = filtered_df['productId'].value_counts()
result = counts[counts > 5]
#print(result.index.tolist())
Final_Result=result.index.tolist()
#print(Final_Result)



output_predicted=Final_Result
#print(output_predicted)
for bst in output_predicted:
    #print(bst)
    mycursor = mydb.cursor()
    sql = "INSERT INTO bestproduct (productid) VALUES (%s)"
    val = [(str(bst))]
    mycursor.execute(sql, val)
    mydb.commit()


