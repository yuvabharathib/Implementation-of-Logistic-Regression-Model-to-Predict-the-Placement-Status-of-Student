# EX 04-Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## Step 1 :
Import the standard libraries such as pandas module to read the corresponding csv file.
## Step 2 :
Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
## Step 3 :
Import LabelEncoder and encode the corresponding dataset values.
## Step 4 :
Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.
## Step 5 :
Predict the values of array using the variable y_pred.
## Step 6 :
Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.
## Step 7 :
Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

## Step 8: 
End the program. 

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Yuvabharathi.B

RegisterNumber: 212222230181
```
import pandas as pd

data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
## HEAD OF THE DATA :
![image](https://github.com/Brindha77/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889143/8a1acdcf-6208-4ccd-a385-dce79acffb23)
## COPY HEAD OF THE DATA:
![image](https://github.com/Brindha77/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889143/a8fbb70d-d081-404a-9852-d8e0cb514d72)
## NULL AND SUM :
![image](https://github.com/Brindha77/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889143/3dc72aad-5424-4b66-af78-4cff6e2b97a6)
## DUPLICATED :
![image](https://github.com/Brindha77/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889143/c3040d19-87ba-41b2-9602-08be1862e318)
## X VALUE:
![image](https://github.com/Brindha77/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889143/65c146b0-3c3d-4e28-9175-575ef0a3e1e9)
## Y VALUE :
![image](https://github.com/Brindha77/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889143/60c983e7-7e30-4915-a582-580814707244)
## PREDICTED VALUES :
![image](https://github.com/Brindha77/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889143/cb2539b7-5d40-4738-aaa6-a59f2d0bc329)
## ACCURACY :
![image](https://github.com/Brindha77/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889143/ebcedcb0-3310-42ca-8122-9b3c9730b289)
## CONFUSION MATRIX :
![image](https://github.com/Brindha77/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889143/deeaec70-cb55-4c5f-b019-bd1a8379d9ce)
## CLASSIFICATION REPORT :
![image](https://github.com/Brindha77/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889143/18a5dede-f1c9-4211-bed3-43d7c14bef9c)
## Prediction of LR :
![image](https://github.com/Brindha77/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118889143/2bcc7f7f-26f7-46a9-ad12-2f6c01497de9)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
