import pandas as pd
import numpy as np
import re


##################################################
import pandas as pd
import numpy as np
from apyori import apriori
#####################################################3


##########################################################################MYSQL CONNECTION##############################################################
import mysql.connector
import csv
mydb = mysql.connector.connect(host="localhost",user="root",password="",database="shopping")

mycursor = mydb.cursor()

mycursor.execute("SELECT id,userId,productId FROM orders")
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
print(merged_df.head())

merged_df.to_csv('transaction.csv', index=False)



##pivot_table=pd.pivot_table(merged_df,values='productName',index='userId',columns='productId',aggfunc='sum')
##
##
##df=pivot_table
##
#######################################################PRINTING ALL THE VALUES########################################################################
##print(data)
##print(data1)
##print(df.head())
##
#######################################################################################################################################################
##
##########################################ASSOCIATION RULE MINING-APRIORI ALGORITHM####################################################################
##
###########################################DATA PREPROCESSING-for using aprori , need to convert data in list format..#################################
##
###########################################CREATING AN EMPTY LIST######################################################################################
##
##transactions = []
##
###print(len(df))
##
##for i in range(0,len(df)):
##    transactions.append([str(df.values[i,j]) for j in range(0,20) if str(df.values[i,j])!='0'])
##
##
#### verifying - by printing the 0th transaction
###print(transactions[0])
##
###print(transactions[1])
############################################################################GENERATE THE RULES###########################################################
##rules = apriori(transactions, min_support=0.003, min_confidance=0.2, min_lift=3, min_length=2)
##rules
############################################################################CONVERTS THE RULES INTO LIST#################################################
##Results = list(rules)
##Results
##
##
##
##
##
#############################################################################convert result in a dataframe for further operation#########################
##df_results = pd.DataFrame(Results)
##pd.set_option('display.max_columns', None)
##pd.set_option('display.max_rows', None)
##print(df_results.head())
####print(df_results['items'])
####for order_list in df_results['items']:
####    order = order_list[0]
####    print(order)
##
##
##
##
##
##
##
##
#############################################################################keep support in a separate data frame so we can use later.###################
##support = df_results.support
##
##
#############################################################################convert orderstatistic in a proper format.order statistic has lhs => rhs as well rhs => lhs we can choose any one for convience.
#############################################################################Let's choose first one which is 'df_results['ordered_statistics'][i][0]'######
##############################################################################all four empty list which will contain lhs, rhs, confidance and lift respectively.###############
##first_values = []
##second_values = []
##third_values = []
##fourth_value = []
##
############################################################################## loop number of rows time and append 1 by 1 value in a separate list..######### 
############################################################################## first and second element was frozenset which need to be converted in list..
##for i in range(df_results.shape[0]):
##    single_list = df_results['ordered_statistics'][i][0]
##    first_values.append(list(single_list[0]))
##    second_values.append(list(single_list[1]))
##    third_values.append(single_list[2])
##    fourth_value.append(single_list[3])
################################################################################ convert all four list into dataframe for further operation..################
##lhs = pd.DataFrame(first_values)
##rhs = pd.DataFrame(second_values)
##
##confidance=pd.DataFrame(third_values,columns=['Confidance'])
##
##lift=pd.DataFrame(fourth_value,columns=['lift'])
##
################################################################################concat all list together in a single dataframe#################################
##df_final = pd.concat([lhs,rhs,support,confidance,lift], axis=1)
##df_final
##############################################################################we have some of place only 1 item in lhs and some place 3 or more so we need to a proper represenation for User to understand. 
############################################################################## replacing none with ' ' and combining three column's in 1 ########################
############################################################################## example : coffee,none,none is converted to coffee, ,##############################
##
##df_final.fillna(value=' ', inplace=True)
##df_final.head()
##
##df_final.head()
#############################################################
##df_final.columns = ['lhs',1,'rhs',2,3,'support','confidance','lift','col_9']
###print(df_final.head())
##
##
##
##
##
##df_final['lhs'] = df_final['lhs'] + str(", ") + df_final[1]
##
##df_final['rhs'] = df_final['rhs']+str(", ")+df_final[2] + str(", ") + df_final[3]
##
##df_final.head()
###############################################################################drop columns 1,2 and 3 because now we already appended to lhs column.###############
##df_final.drop(columns=[1,2,3],inplace=True)
##
##df_final.head()
##print(df_final.head())
###############################################################################this is final output. You can sort based on the support lift and confidance#########
###print(df_final.sort_values('lift', ascending=False).head(10))
####var.replace('',np.nan,inplace=True)
####
####
##print(df_final['lhs'])
##
##for order_list in df_final['lhs']:
##    order = order_list[0]
##    print(order)
##
##
##
##print(df_final['rhs'])
