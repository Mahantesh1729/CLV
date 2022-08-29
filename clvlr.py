import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# Importing the dataset
dataset = pd.read_excel(r'Dataset.xlsx')
dataset.head()

a1={'L': 0,'M': 1,'H':2}
dataset['CLV_TYPE'] = dataset['CLV_TYPE'].map(a1)

X = dataset[['Bank_Balance','(Average)Interest Rate Margin(%)','Loan_Interest','Fees_Service','Total_Earning','Retain_Amount','Service_Spent','Discount_Rate','Discount_Amount','Gross_Margin','Retention_rate']]

Y = dataset[['CLV_TYPE']] 
dataset.head()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20 ,random_state = 0)

classifier = linear_model.LogisticRegression()
m=classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

cm = confusion_matrix(Y_test, y_pred)
print(cm)

print("Accuracy of Logistic Regression Model is:",
     metrics.accuracy_score(Y_test,y_pred)*100,"%")
X.head(5)

a=float(input("Enter The Customer Bank Balance:"))
b=float(input("Enter The (Average)Interest Rate Margin(%) :"))
c=float(input("Loan_Interest :"))
d=float(input("Fees_Service :"))
e=a+b+c+d
f=float(input("Retain_Amount :"))
g=float(input("Service_Spent :"))
h=float(input("Discount_Rate :"))
i=float(input("Discount_Amount :"))
j=e-f-g-h-i
k=float(input("Retention_rate :"))
print("The Predicted Value of CLV is :",classifier.predict([[a,b,c,d,e,f,g,h,i,j,k]]))
f=m.predict([[a,b,c,d,e,f,g,h,i,j,k]])
if f==0:
  print("The CLV is Low")
elif f==1:
  print("The CLV is Medium")
else:
  print("The CLV is High")