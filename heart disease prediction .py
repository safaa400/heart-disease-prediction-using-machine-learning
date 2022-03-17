import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib



#importing dataset
data=pd.read_csv('heart.csv')
#print(data.head())


#handle missing values
#print(data.isnull().sum())


#handling duplicate data
check_dup =data.duplicated().any()
##print(check_dup) >> true
data =data.drop_duplicates()
check_dup1=data.duplicated().any()
#print(check_dup1)>> false


#data processing >> columns features into  numerical data || categorial data

categorial=[]
numerical =[]

for column in data.columns:
    if data[column].nunique() <= 10:
        categorial.append(column)
    else:
        numerical.append(column)

#print(categorial,numerical) >> ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target'] ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


#encoding categorial data

#print(data['cp'].unique())

#because sex and target already 0 1
categorial.remove('sex')
categorial.remove('target')
#print(categorial)

#data manupulation
data=pd.get_dummies(data,columns=categorial,drop_first=True)
#print(data.head())

#feature scaling  >> for numerical colums ,machine learning that doesn't require feature scaling is non-linear *not distance-based*
#making values of the columns in the same range || -4 to 4  | -3 to 3 etc

st=StandardScaler()
data[numerical]=st.fit_transform(data[numerical])
#print(data[numerical])


#splitting data into training and testing
x=data.drop('target',axis=1)
y=data['target']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)


#-------------------------------------------------------------------linear classification ---------------------------------------------------------------

#-------------------------------------------------------------------logistic regression ---------------------------------------------------------------
log=LogisticRegression()
log.fit(x_train,y_train)

y_predict1 = log.predict(x_test)
accuracy1=accuracy_score(y_test,y_predict1)
print("logistic regression result ", accuracy1)

#-----------------------------------------------------------------------------SVM--------------------------------------------------------
svm=svm.SVC()
svm.fit(x_train,y_train)
y_predict2=svm.predict(x_test)
accuracy2=accuracy_score(y_test,y_predict2)
print("svm result",accuracy2)
#-----------------------------------------------------------------------------KNN--------------------------------------------------------


knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_predict3=knn.predict(x_test)
accuracy3=accuracy_score(y_test,y_predict3)
print("knn result",accuracy3)
#-----------------------------------------------------------------------------knn with multiple k--------------------------------------------------------


score=[]
for k in range(1,40):
    knn1 = KNeighborsClassifier(n_neighbors=k)
    knn1.fit(x_train,y_train)
    y_predict4=knn1.predict(x_test)
    accuracy4=accuracy_score(y_test,y_predict4)
    score.append(accuracy4)

#print( "knn multiple k's",score)

#-----------------------------------------------------------------------------KNN n=2--------------------------------------------------------

knn2 = KNeighborsClassifier(n_neighbors=2)
knn2.fit(x_train,y_train)
y_predict5=knn2.predict(x_test)
accuracy5=accuracy_score(y_test,y_predict5)
print("knn with k=2",accuracy5)

#-----------------------------------------------------------------non-linear models--------------------------------------------------------------------

data1=pd.read_csv('heart1.csv')
data1=data1.drop_duplicates()
x1=data1.drop('target',axis=1)
y1=data1['target']
x1_train, x1_test ,y1_train ,y1_test =train_test_split(x1,y1,test_size=0.2,random_state=42)

#-----------------------------------------------------------------Decision tree--------------------------------------------------------------------
dt=DecisionTreeClassifier()
dt.fit(x1_train,y1_train)
y1_predict=dt.predict(x1_test)
accuracy6=accuracy_score(y1_test,y1_predict)
print("decision tree result",accuracy6)
#-----------------------------------------------------------------random forest--------------------------------------------------------------------

rf=RandomForestClassifier()
rf.fit(x1_train,y1_train)
y1_predict1=rf.predict(x1_test)
accuracy7=accuracy_score(y1_test,y1_predict1)
print("random forest result",accuracy7)

#-----------------------------------------------------------------Gradient boosting--------------------------------------------------------------------

gb=GradientBoostingClassifier()
gb.fit(x1_train,y1_train)
y1_predict2=gb.predict(x1_test)
accuracy8=accuracy_score(y1_test,y1_predict2)
print("gradient boosting result",accuracy8)
#-----------------------------------------------------------------comparing --------------------------------------------------------------------


final_data=pd.DataFrame({'Models':['LR','SVM','KNN','DT','RF','GB'],'accuracy':[
    accuracy_score(y_test,y_predict1),
    accuracy_score(y_test,y_predict2),
    accuracy_score(y_test,y_predict5),
    accuracy_score(y1_test,y1_predict),
    accuracy_score(y1_test,y1_predict1),
    accuracy_score(y1_test,y1_predict2)


]})
print(final_data)
sns.barplot(final_data['Models'],final_data['accuracy'])
#-----------------------------------------------------------------predict new data with random forest model --------------------------------------------------------------------
#age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target

new_data=pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
    'slope':2,
    'ca':2,
    'thal':3
},index=[0])
print('new data entered' ,'\n' ,new_data)

rf.fit(x1,y1)
p=rf.predict(new_data)
if p[0]==0:
    print("prediction for new data is :no disease")
else:
    print("prediction for new data is :heart disease detected")

#-----------------------------------------------------------------saving the model --------------------------------------------------------------------
joblib.dump(rf,'detact_heart_disease_model')

#in the future to load the model
#model=joblib.load('detact_heart_disease_model')
#model.predict(new_data)


plt.show()










