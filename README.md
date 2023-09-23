# EXP 2: Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### STEP 1:
Import the needed packages
### STEP 2: 
Assigning hours To X and Scores to Y
### STEP 3:
Plot the scatter plot
### STEP 4 :
Use mse,rmse,mae formmula to find
## Program:
Developed by: Yuvabharathi.B

RegisterNumber:212222230181

Program to implement the simple linear regression model for predicting the marks scored.
### df.head()
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
```
### df.tail()
```
df.tail()
```
### Array value of X
```
X = df.iloc[:,:-1].values
X
```
### Array value of Y
```
Y = df.iloc[:,1].values
Y
```
### Values of Y prediction
```
Y_pred
```
### Array values of Y test
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
```
### Training Set Graph
```
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
### Test Set Graph
```
plt.scatter(X_test,Y_test,color="grey")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
### Values of MSE,MAE AND RMSE
```
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output:
### df.head()
![l1](https://github.com/Brindha77/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118889143/8c8dd58a-a73b-4005-ba81-1d8a13367284)
### df.tail()
![l2](https://github.com/Brindha77/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118889143/556ed15a-c648-4a2c-a31d-2c3f16f8739b)
### Array value of X
![l3](https://github.com/Brindha77/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118889143/fc44abb6-91f7-4201-a308-0439b343b4c3)
### Array value of Y
![l4](https://github.com/Brindha77/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118889143/d02ec9fa-0030-45c0-abab-98b58141838d)
### Values of Y prediction
![l5](https://github.com/Brindha77/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118889143/e1f2a72a-48fb-4a30-94df-609cdf01f7dd)
### Array values of Y test
![l8](https://github.com/Brindha77/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118889143/4fa40811-e4ee-41ac-bb7c-9af4b62c2ccb)
### Training Set Graph
![l6](https://github.com/Brindha77/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118889143/97df94c5-1dee-4a9e-9aed-518f3e7a22b6)
### Test Set Graph
![l7](https://github.com/Brindha77/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118889143/a5daf700-7e24-4204-9be8-da6f7249e832)
### Values of MSE,MAE AND RMSE
![l9](https://github.com/Brindha77/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118889143/2d7e0876-1d50-4848-8222-b9c7ddbf0f9d)
![l10](https://github.com/Brindha77/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118889143/855c2324-6cd7-418c-b488-0ac23e3be308)
![l11](https://github.com/Brindha77/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118889143/e7e90d2e-ebfa-4b8b-ad4d-393910849040)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
