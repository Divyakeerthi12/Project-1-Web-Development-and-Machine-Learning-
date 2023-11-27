import pandas as pd
import numpy as np
import re





import mysql.connector
import csv
mydb = mysql.connector.connect(host="localhost",user="root",password="",database="shopping")

mycursor = mydb.cursor()





mycursor.execute("SELECT * FROM orders")
rows = mycursor.fetchall()
column_names = [i[0] for i in mycursor.description]
fp = open('transaction.csv', 'w')
myFile = csv.writer(fp, lineterminator = '\n')
myFile.writerow(column_names)   
myFile.writerows(rows)
fp.close()
