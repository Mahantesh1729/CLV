import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib

input_file = "Dataset.xlsx"
df = pd.read_excel(input_file, header = 0)
df.head()

a={"L":0, "M":1, "H":2}
df['CLV_TYPE'] = df['CLV_TYPE'].map(a)
df.head()

y = df["CLV_TYPE"]
X = df[['Bank_Balance','(Average)Interest Rate Margin(%)','Loan_Interest','Fees_Service','Total_Earning','Retain_Amount','Service_Spent','Discount_Rate','Discount_Amount','Gross_Margin','Retention_rate']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.70)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, random_state = 30)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("The Confusion Matrix is :")
print(cm)
from sklearn import metrics
print("Accuracy of Random Forest Model is:",
     metrics.accuracy_score(y_test,y_pred)*100,"%")
X.head()

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=10)
clf = clf.fit(X, y)
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
print("The Predicted Value of CLV is :",clf.predict([[a,b,c,d,e,f,g,h,i,j,k]]))
f=clf.predict([[a,b,c,d,e,f,g,h,i,j,k]])
# f=clf.predict([[200000,3.15,615.75,5000,2500,2000,0.1,12000,77115.75,0.478]])
if f==0:
  print("The CLV is Low")
elif f==1:
  print("The CLV is Medium")
else:
  print("The CLV is High")
  
df1=df[['Age','CLV']]
bins =  np.arange(20,50,5)
groups = df1.groupby(pd.cut(df1['Age'],bins)).mean()
print(groups.head())
groups['CLV'].plot.line()