#Importing Libraries
import pandas as pd
import numpy as np
import sklearn 
import matplotlib.pyplot as plt
import pickle


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Importing Data

df = pd.read_csv("iris.csv")

# X and y
X= df.drop("variety",axis =1)
y= df['variety']


# Splitting into Train Test Set

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# Create a model with feature names

model = tree.DecisionTreeClassifier()
model.fit(X,y)

#Make Pickle File of our model

pickle.dump(model, open("model.pkl","wb"))
