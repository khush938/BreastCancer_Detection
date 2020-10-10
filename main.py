import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

#importing the dataset 
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:31].values
Y = dataset.iloc[:, 31].values
#print(X)
dataset.head()

print("Cancer data set dimensions : {}".format(dataset.shape))

dataset.groupby('diagnosis').size()

#Visualization of data
dataset.groupby('diagnosis').hist(figsize=(12, 12))

dataset.isnull().sum()
dataset.isna().sum()

dataframe = pd.DataFrame(Y)
#Encoding categorical data values 
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(dataset.iloc[:,1].values)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#print(X)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_test)
X_test = sc.transform(X_test)
print(X_test[0])
# #Fitting the Logistic Regression Algorithm to the Training Set
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train, Y_train)
# #95.8 Acuracy

# #Fitting K-NN Algorithm
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# classifier.fit(X_train, Y_train)
# #95.1 Acuracy

# #Fitting SVM
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'linear', random_state = 0)
# classifier.fit(X_train, Y_train) 
# #97.2 Acuracy

# #Fitting K-SVM
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0)
# classifier.fit(X_train, Y_train)
# #96.5 Acuracy

# #Fitting Naive_Bayes
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, Y_train)
# #91.6 Acuracy

# #Fitting Decision Tree Algorithm
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, Y_train)
# #95.8 Acuracy

#Fitting Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
#print(X_train)
print('Random Forest Classifier Training Accuracy:', classifier.score(X_train, Y_train))
joblib.dump(classifier, "./classifier.joblib")
#print(type(X_test))
#kj = X_test[0]

predictions = classifier.predict(X_test)
print(predictions)
#98.6 Acuracy

#predicting the Test set results
# print(X_test)
# Y_pred = classifier.predict(X_test)
# print(Y_pred)
# #Creating the confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(Y_test, Y_pred)
# print(cm)