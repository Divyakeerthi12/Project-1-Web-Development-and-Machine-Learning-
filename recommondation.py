##import pandas as pd
##import numpy as np
##import re
##import csv
##
##from sklearn.model_selection import train_test_split
##from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
##from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
##
##import keras
##from keras.models import Sequential,load_model
##from keras import layers
##from keras.layers import LSTM,Dense,Dropout,Conv1D,BatchNormalization,MaxPooling1D,Flatten
##import pickle
##
##import matplotlib.pyplot as plt
##
##from collections import Counter
##import warnings
##warnings.filterwarnings('ignore')
##
##import mysql.connector
##import csv
##mydb = mysql.connector.connect(host="localhost",user="root",password="",database="shopping")
##
##mycursor = mydb.cursor()
##
##mycursor.execute("SELECT * FROM recommended")
##
##rows = mycursor.fetchall()
##column_names = [i[0] for i in mycursor.description]
##fp = open('traincnn.csv', 'w')
##myFile = csv.writer(fp, lineterminator = '\n')
##myFile.writerow(column_names)   
##myFile.writerows(rows)
##fp.close()
##
##
##def clean_data(data):
##    data.replace('',np.nan,inplace=True)
##    data.dropna(axis=0, how='any', inplace=True)
##    data.to_csv('traincnn.csv', index=False)
##    return data
##
##data = pd.read_csv('traincnn.csv', header=0, index_col=False, delimiter=',')
##data = clean_data(data)
##print(data)
##
##
##
##
#######  LSTM
##
##genre_list = data.iloc[:, -1]
##encoder = LabelEncoder()
##y = encoder.fit_transform(genre_list)
##scaler = StandardScaler()
##X = scaler.fit_transform(np.array(data.iloc[:,:-1], dtype = float))
##x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
##
##x_train = x_train.reshape(x_train.shape[0],-1,1)
##x_test = x_test.reshape(x_test.shape[0],-1, 1)
##y_train = keras.utils.to_categorical(y_train,2)
##y_test = keras.utils.to_categorical(y_test,2)
##
##model = Sequential()
##model.add(LSTM(256, input_shape=x_train.shape[1:]))
##model.add(Dropout(0.5))
##model.add(Dense(128,activation='relu'))
##model.add(Dense(64, activation='relu'))
##model.add(Dense(2,activation='softmax'))
##model.summary()
##model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
##history=model.fit(x_train,y_train,batch_size=32,epochs=100,verbose=1,validation_data=(x_test, y_test))
##model.save("trainlstm.h5")
##f = open('lstm_history.pckl', 'wb')
##pickle.dump(history.history, f)
##f.close()
##
##
########   CNN  ###############
##
####genre_list = data.iloc[:, -1]
####encoder = LabelEncoder()
####y = encoder.fit_transform(genre_list)
####print(y)
####scaler = StandardScaler()
####X = scaler.fit_transform(np.array(data.iloc[:,:-1], dtype = float))
####x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
####
####x_train = x_train.reshape(x_train.shape[0],-1,1)
####x_test = x_test.reshape(x_test.shape[0],-1, 1)
####y_train = keras.utils.to_categorical(y_train,2)
####y_test = keras.utils.to_categorical(y_test,2)
####
####print(x_train.shape[1:])
####
####model = Sequential()
####model.add(Conv1D(filters=6, kernel_size=21, strides=1, padding='same', activation='relu',input_shape=x_train.shape[1:]))
####model.add(BatchNormalization()) 
####model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
####model.add(Conv1D(filters=16, kernel_size=5, strides=1, padding='same',activation='relu'))
####model.add(BatchNormalization())
####model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
####model.add(Flatten())
####model.add(Dense(120, activation='relu'))
####model.add(Dense(84))
####model.add(Dense(2, activation='softmax'))
####model.summary()
####
####model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
####model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.SpecificityAtSensitivity(0.5), keras.metrics.SensitivityAtSpecificity(0.5), 'accuracy'])
####history=model.fit(x_train,y_train,batch_size=32,epochs=100,verbose=1,validation_data=(x_test, y_test))
####model.save("traincnn.h5")
####f = open('cnn_history.pckl', 'wb')
####pickle.dump(history.history, f)
####f.close()
##
################ GRAPH #######################3
##f = open('lstm_history.pckl', 'rb')
##history = pickle.load(f)
##f.close()
##print(history)
##plt.figure(0)
##plt.plot(history['accuracy'], label='training accuracy')
##plt.plot(history['val_accuracy'], label='val accuracy')
##plt.title('ACCURACY')
##plt.xlabel('EPOCHS')
##plt.ylabel('ACCURACY')
##plt.legend()
##plt.show()
##plt.figure(0)
##plt.plot(history['specificity_at_sensitivity'],label='SPECIFICITY')
##plt.title('SPECIFICITY')
##plt.xlabel('EPOCHS')
##plt.ylabel('SPECIFICITY')
##plt.legend()
##plt.show()
##plt.figure(0)
##plt.plot(history['sensitivity_at_specificity'], label='SENSITIVITY')
##plt.title('SENSITIVITY')
##plt.xlabel('EPOCHS')
##plt.ylabel('SENSITIVITY')
##plt.legend()
##plt.show()
##plt.figure(0)
##plt.plot(history['accuracy'], label='F1SCORE')
##plt.title('F1SCORE')
##plt.xlabel('EPOCHS')
##plt.ylabel('F1SCORE')
##plt.legend()
##plt.show()
##
##
#########################################################################################
##
##def search_csv(filename, search_column, search_value):
##    with open(filename, 'r') as file:
##        csv_reader = csv.reader(file)
##        headers = next(csv_reader)  # Read and store the header row
##        search_column_index = headers.index(search_column)
##        matching_rows = []
##
##        for row in csv_reader:
##            if row[search_column_index] == search_value:
##                matching_rows.append(row)
##
##        return matching_rows
##
##
##
##
##mycursor.execute("SELECT userid FROM loginuser")
##rows = mycursor.fetchall()
##uid=str(rows)
##uid=uid.replace('[(','')
##uid=uid.replace(',)]','')
##uid=uid.replace("'","")
##print(uid)
##uidd = list(uid.split(" "))
##
##mycursor.execute("SELECT productId FROM orders where userId=%s order by id desc",(uidd))
##rows1 = mycursor.fetchall()
##
##pid=str(rows1)
##pid=pid.replace('[(','')
##pid=pid.replace(',)]','')
##pid=pid.replace("'","")
##pid=pid.replace(",)","")
##pid=pid.replace("(","")
##pid=pid.replace(" ","")
##pid=pid.replace(","," ")
##pid = list(pid.split(" "))
##print(pid)
##
##
##for x in pid:
##  search_result = search_csv('traincnn.csv', 'productId', str(x))
##  ###dd=data.loc[[15],:]
##  df = pd.DataFrame(search_result,columns=['id','userId','productId','status'])
##  print(df)
##  genlist = df.iloc[:, -1]
##  encoder = LabelEncoder()
##  yd= encoder.fit_transform(genlist)
##  #print(yd)
##  scaler = StandardScaler()
##  xtest = scaler.fit_transform(np.array(df.iloc[:,:-1], dtype = float))
##  model = load_model('traincnn.h5')
##  pred = model.predict(xtest,verbose=0)
##  classes = np.argmax(pred, axis=-1)
##  print(classes)
##  if len(classes)==1:
##      prd=str(classes)
##      prd=prd.replace('[','')
##      prd=prd.replace(']','')
##      print(prd)
##      mycursor = mydb.cursor()
##      sql = "INSERT INTO predicted (userid,productid,status) VALUES (%s,%s,%s)"
##      val = [(str(uid)),str(x),prd]
##      mycursor.execute(sql, val)
##      mydb.commit()
##
##  elif len(classes)==2:
##      prd=str(classes[0])
##      prd=prd.replace('[','')
##      prd=prd.replace(']','')
##      print(prd)
##      mycursor = mydb.cursor()
##      sql = "INSERT INTO predicted (userid,productid,status) VALUES (%s,%s,%s)"
##      val = [(str(uid)),str(x),prd]
##      mycursor.execute(sql, val)
##      mydb.commit()
##  else:
##      gh=max(k for k,v in Counter(classes).items() if v>1)
##      prd=str(gh)
##      print(prd)
##      
##
