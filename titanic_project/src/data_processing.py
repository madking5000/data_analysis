import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


from sklearn.model_selection import train_test_split


def data_preprocess(data):
    df = data
    #remove the column with many missing features 
    df = df.drop(columns = ['PassengerId', 'Cabin', 'Ticket', 'Name'])
    #df fill missing values 
    df.fillna({'Embarked': 'S'}, inplace=True)

    #df map the sex with 1 and zero 
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    #encode the embarked 
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    #fill the age of missing values with middle age 
    df['Age'] = df['Age'].fillna(df['Age'].median())

    #calculate the family sise 
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    #split into features and target variables 

    X = df.drop(columns='Survived')
    y = df['Survived']

    #split into train and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test