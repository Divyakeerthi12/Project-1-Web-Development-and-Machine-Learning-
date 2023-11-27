import pandas as pd
import csv
import numpy as np
import re


##################################################
import pandas as pd
import numpy as np
from apyori import apriori
#####################################################3


##########################################################################MYSQL CONNECTION##############################################################
import mysql.connector
mydb = mysql.connector.connect(host="localhost",user="root",password="",database="shopping")

mycursor = mydb.cursor()

mycursor.execute("SELECT id,userId,productId,invoicenum FROM orders")
rows = mycursor.fetchall()
column_names = [i[0] for i in mycursor.description]
fp = open('assoc.csv', 'w')
myFile = csv.writer(fp, lineterminator = '\n')
myFile.writerow(column_names)   
myFile.writerows(rows)
fp.close()

def clean_data(data):
    data.replace('',np.nan,inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    data.to_csv('assoc.csv', index=False)
    return data


data = pd.read_csv('assoc.csv', header=0, index_col=False, delimiter=',')
data = clean_data(data)




mycursor.execute("SELECT id,productName,category,subCategory FROM products")
rows = mycursor.fetchall()
column_names = [i[0] for i in mycursor.description]
fp = open('merge.csv', 'w')
myFile = csv.writer(fp, lineterminator = '\n')
myFile.writerow(column_names)   
myFile.writerows(rows)
fp.close()



def clean_data(data):
    data.replace('',np.nan,inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    data.to_csv('merge.csv', index=False)
    return data
data1 = pd.read_csv('merge.csv', header=0, index_col=False, delimiter=',')
data1 = clean_data(data1)


data1.rename(columns={'id':'productId'},inplace=True)


merged_df=pd.merge(data,data1,on='productId')
#print(merged_df.head())

merged_df = merged_df.sort_values('invoicenum')
columns_to_drop = ['category', 'subCategory','id']
merged_df = merged_df.drop(columns_to_drop, axis=1)
merged_df.to_csv('transaction.csv', index=False)






##############################################################################APPLYING ASSOCIATION RULE MINING###############################################################################################
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('transaction.csv')
print(df.head())



#Drop a column by specifying its name





### Rename a column
##df.rename(columns={'id': 'InvoiceNo'}, inplace=True)
##
### Display the updated dataframe
##print(df)





df['productName'] = df['productName'].str.strip()
df.dropna(axis=0, subset=['invoicenum'], inplace=True)
#df['invoicenum'] = df['invoicenum'].astype('str')
#df = df[~df['invoicenum'].str.contains('C')]

print(df)


basket = (df.groupby(['invoicenum', 'productName'])['productId'] .sum().unstack().reset_index().fillna(0).set_index('invoicenum'))

print(basket)

def encode_units(x):
    if x <= 0:
        return False
    if x >= 1:
        return True

basket_sets = basket.applymap(encode_units)
print(basket_sets)
frequent_itemsets = apriori(basket_sets,min_support=0.07,use_colnames=True)
print(frequent_itemsets)

rules = association_rules(frequent_itemsets,metric="lift",min_threshold=1)
print(rules)

rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
ants=rules["antecedents"].tolist()
##antsres=",".join([item for item in ants if isinstance(item,str)])
##print(antsres)
##quoted_list=['"{}"'.format(item) for item in antsres]
##result=",".join(quoted_list)
##print(result)

rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
consq=rules["consequents"].tolist()
consqres=','.join([item1 for item1 in consq if isinstance(item1,str)])








#####################################################################################DATABASE PART OF EXTRACTING ANTECEDENT AND CONSEQUENT#######################################################################


sql = "INSERT INTO association (antecedent,consequent) VALUES (%s,%s)"
val = [(str(ants)),(str(consq))]
mycursor.execute(sql, val)
mydb.commit()


##
##for item2,item3 in zip(antsres,consqres):
##    #print(bst)
##    mycursor = mydb.cursor()
##    sql = "INSERT INTO association (antecedent,consequent) VALUES (%s,%s)"
##    val = [(str(ants)),(str(consq))]
##    mycursor.execute(sql, val)
##    mydb.commit()


for item2,item3 in zip(ants,consq):
    print(item2)
    mycursor = mydb.cursor()
    sql = "INSERT INTO association (antecedent,consequent) VALUES (%s,%s)"
    val = [(str(item2)),(str(item3))]
    mycursor.execute(sql, val)
    mydb.commit()

