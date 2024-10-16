# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries. 
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy. 
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: THEJASWINI D
RegisterNumber:212223110059
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
```
![EXP8-1](https://github.com/user-attachments/assets/4f7775aa-c63d-4b8e-af78-9ad7d65e1717)
```
data.info()
data.isnull().sum()
data["left"].value_counts()
```
![EXP8-2](https://github.com/user-attachments/assets/c370db4e-6695-4918-be9a-f3ce466067aa)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
![EXP8-3](https://github.com/user-attachments/assets/b7cad503-796e-4b3c-b4ce-548ae379c947)
```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
![EXP8-4](https://github.com/user-attachments/assets/40ec4f84-c4e5-4528-aea6-0f59601b307b)
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![EXP8-5](https://github.com/user-attachments/assets/49ba6901-2354-4c7f-a068-72eb48be24f9)
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
## Output:

![EXP8-OUT](https://github.com/user-attachments/assets/29532cf4-6ae7-4e86-864b-3992877f46d4)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
