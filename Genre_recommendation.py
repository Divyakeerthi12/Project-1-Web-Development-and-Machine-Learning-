import pandas as pd
import numpy as np
import csv
import mysql.connector
mydb = mysql.connector.connect(host="localhost",user="root",password="",database="shopping")

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM recommended")
rows = mycursor.fetchall()
column_names = [i[0] for i in mycursor.description]
fp = open('recomm.csv', 'w')
myFile = csv.writer(fp, lineterminator = '\n')
myFile.writerow(column_names)   
myFile.writerows(rows)
fp.close()

def clean_data(data):
    data.replace('',np.nan,inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    data.to_csv('recomm.csv', index=False)
    return data
data = pd.read_csv('recomm.csv', header=0, index_col=False, delimiter=',')
data = clean_data(data)
print(data.head())

mycursor.execute("SELECT id,category,subCategory,productName FROM products")
rows = mycursor.fetchall()
column_names = [i[0] for i in mycursor.description]
fp = open('green.csv', 'w')
myFile = csv.writer(fp, lineterminator = '\n')
myFile.writerow(column_names)   
myFile.writerows(rows)
fp.close()

def clean_data(data):
    data.replace('',np.nan,inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    data.to_csv('green.csv', index=False)
    return data

data1 = pd.read_csv('green.csv', header=0, index_col=False, delimiter=',')
data1 = clean_data(data1)
print(data1.head())
#################################################################################################To get books_df##############################################################################
books_df=data1
print(books_df.head())

indices_to_dropp = books_df[books_df['category'].isin([6,5,4, 2, 1])].index
books_df = books_df.drop(indices_to_dropp)
subcategory_to_genree={14:"Novel",15:"Horror",8:"Comics"}
books_df["genre"] = books_df["subCategory"].map(subcategory_to_genree)
books_df.rename(columns={'id': 'book_id','productName':'Title'}, inplace=True)
print(books_df.head())



################################################################################################################################################################################################
data1.rename(columns={'id': 'productId'}, inplace=True)
print(data1.head())

merged_dff = pd.merge(data, data1, on='productId', how='inner')
print(merged_dff.head())

indices_to_drop = merged_dff[merged_dff['category'].isin([4, 2, 1])].index
merged_dff = merged_dff.drop(indices_to_drop)
pd.set_option('display.max_columns', None)
print(merged_dff.head())

subcategory_to_genre={14:"Novel",15:"Horror",8:"Comics"}

merged_dff["genre"] = merged_dff["subCategory"].map(subcategory_to_genre)
print(merged_dff.head())


preferences_dict = merged_dff.set_index('userId')['genre'].to_dict()
print(preferences_dict)

###############################################################################################################################################################################################


def generate_recommendations(user_id, preference):
    if user_id in preferences_dict:
        user_preference = preferences_dict[user_id]
    else:
        print(f"User {user_id} preferences not found. Using generic recommendations.")
        user_preference = None

    if preference == "Same" and user_preference is not None:
        # Recommend books from the same genre as the user preference
        recommendations = books_df[books_df['genre'] == user_preference]['Title'].tolist()
    elif preference == "Different" and user_preference is not None:
        # Recommend books from different genres than the user preference
        recommendations = books_df[books_df['genre'] != user_preference]['Title'].tolist()
    else:
        # If preference is not provided or user preference not found, provide generic recommendations
        recommendations = books_df['Title'].tolist()

    return recommendations




##################################################################################################################################################################################################


# Example usage
for value in merged_dff['userId']:
    #print(f"{userId}: {value}")
    user_id = value
    column_name = 'status'  # Replace 'column_name' with the actual name of the column

# Fetch the value based on the userid
    value_for_userid = merged_dff.loc[merged_dff['userId'] == user_id, column_name].values[0]

# Print the value or use it as per your requirement
    print("Value for userid {}: {}".format(user_id, value_for_userid))

    preference = value_for_userid
    recommendations = generate_recommendations(user_id, preference)
    print(f"Recommended books for User {user_id} with preference '{preference}' genre:")
    print(recommendations)
    




