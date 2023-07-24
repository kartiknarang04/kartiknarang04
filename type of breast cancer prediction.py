
#import libraries
import numpy as np
import sklearn.datasets

#import datasets
breast_cancer=sklearn.datasets.load_breast_cancer()
print(breast_cancer)
X=breast_cancer.data
Y=breast_cancer.target
print(X)
print(Y)
print(X.shape,Y.shape)

#import datas to pandas
import pandas as pd

data=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
data['class']=breast_cancer.target
data.head()

#get idea of type of data
data.describe()

#class 1: beningn  
#class 0: malignant
print(data['class'].value_counts())
print(breast_cancer.target_names)

#mean of data by class
data.groupby('class').mean()

#train and test split
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
print(Y.shape,Y_train.shape,Y_test.shape)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
print(Y.mean(),Y_train.mean(),Y_test.mean())
print(X.mean(),X_train.mean(),X_test.mean())
#test size to specify the percentage of test data needed
#stratify for correct distribution of data
#random state is used specifically split the data each value of the random state split


from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression() #loading the model

#training the model
classifier.fit(X_train,Y_train)

#evaluating the model
from sklearn.metrics import accuracy_score

prediction_on_training_data=classifier.predict(X_train)
accuracy_train=accuracy_score(Y_train,prediction_on_training_data)
print(accuracy_train)

#prediction on training data
prediction_on_test_data=classifier.predict(X_test)
accuracy_test=accuracy_score(Y_test,prediction_on_test_data)
print(accuracy_test)

#reading the csv files using pandas
import pandas as pd

path="/content/drive/MyDrive/perfect learning/breast cancer.csv"
df=pd.read_csv(path)
rmean=df["radius_mean"].tolist()
tmean=df["texture_mean"].tolist()
pmean=df["perimeter_mean"].tolist()
amean=df["area_mean"].tolist()
smean=df["smoothness_mean"].tolist()
compactmean=df["compactness_mean"].tolist()
concavemean=df["concavity_mean"].tolist()
conmean=df["concave points_mean"].tolist()
symean=df["symmetry_mean"].tolist()
fmean=df["fractal_dimension_mean"].tolist()
r2mean=df["radius_se"].tolist()
t2mean=df["texture_se"].tolist()
p2mean=df["perimeter_se"].tolist()
a2mean=df["area_se"].tolist()
s2mean=df["smoothness_se"].tolist()
compact2mean=df["compactness_se"].tolist()
concave2mean=df["concavity_se"].tolist()
con2mean=df["concave points_se"].tolist()
sy2mean=df["symmetry_se"].tolist()
f2mean=df["fractal_dimension_se"].tolist()
r3mean=df["radius_worst"].tolist()
t3mean=df["texture_worst"].tolist()
p3mean=df["perimeter_worst"].tolist()
a3mean=df["area_worst"].tolist()
s3mean=df["smoothness_worst"].tolist()
compact3mean=df["compactness_worst"].tolist()
concave3mean=df["concavity_worst"].tolist()
con3mean=df["concave points_worst"].tolist()
sy3mean=df["symmetry_worst"].tolist()
f3mean=df["fractal_dimension_worst"].tolist()
input_data=[]
finalinput=[]
for i in range(0,len(rmean)):
  input_data.append([])
  finalinput.append([])
  input_data[i].append(rmean[i])
  input_data[i].append(tmean[i])
  input_data[i].append(pmean[i])
  input_data[i].append(amean[i])
  input_data[i].append(smean[i])
  input_data[i].append(compactmean[i])
  input_data[i].append(concavemean[i])
  input_data[i].append(conmean[i])
  input_data[i].append(symean[i])
  input_data[i].append(fmean[i])
  input_data[i].append(r2mean[i])
  input_data[i].append(t2mean[i])
  input_data[i].append(p2mean[i])
  input_data[i].append(a2mean[i])
  input_data[i].append(s2mean[i])
  input_data[i].append(compact2mean[i])
  input_data[i].append(concave2mean[i])
  input_data[i].append(con2mean[i])
  input_data[i].append(sy2mean[i])
  input_data[i].append(f2mean[i])
  input_data[i].append(r3mean[i])
  input_data[i].append(t3mean[i])
  input_data[i].append(p3mean[i])
  input_data[i].append(a3mean[i])
  input_data[i].append(s3mean[i])
  input_data[i].append(compact3mean[i])
  input_data[i].append(concave3mean[i])
  input_data[i].append(con3mean[i])
  input_data[i].append(sy3mean[i])
  input_data[i].append(f3mean[i])
  numpyinput=np.asarray(input_data[i])
  numpyinput.reshape(1,-1)
  finalinput[i].append(numpyinput)

n1=int(input("enter staring number: "))
n2=int(input("enter ending number: "))
for i in range(n1,n2):
  prediction=classifier.predict(finalinput[i])
  if prediction==0:
    print("Cancer is Malignant")
  else:
    print("Cancer is Benign")
