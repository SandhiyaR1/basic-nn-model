# Developing a Neural Network Regression Model

### AIM
To develop a neural network regression model for the given dataset.

### THEORY
Explain the problem statement

### Neural Network Model
Include the neural network model diagram.

### DESIGN STEPS
- STEP 1:Loading the dataset
- STEP 2:Split the dataset into training and testing
- STEP 3:Create MinMaxScalar objects ,fit the model and transform the data.
- STEP 4:Build the Neural Network Model and compile the model.
- STEP 5:Train the model with the training data.
- STEP 6:Plot the performance plot
- STEP 7:Evaluate the model with the testing data.

### PROGRAM
### Name:SANDHIYA R
### Register Number:212222230129
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)
worksheet=gc.open("ex1DL").sheet1
df=worksheet.get_all_values()
print(df)
ds1=pd.DataFrame(df[1:],columns=df[0])
ds1=ds1.astype({'input':'float'})
ds1=ds1.astype({'output':'float'})
ds1.head()
x = ds1[['input']].values
y = ds1[['output']].values
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33,random_state=33)
scaler=MinMaxScaler()
scaler.fit(x_train)
xtrain=scaler.transform(x_train)
model=Sequential([Dense(8,activation="relu",input_shape=[1]),Dense(10,activation="relu"),Dense(1)])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(xtrain,y_train,epochs=2000)
cf=pd.DataFrame(model.history.history)
cf.plot()
xtrain=scaler.transform(x_test)
model.evaluate(xtrain,y_test)
n=[[17]]
n=scaler.transform(n)
model.predict(n)
```
### Dataset Information
#### DATASET.HEAD():
![image](https://github.com/user-attachments/assets/ec7b839f-d7e4-4216-a11b-11588541a75e)
#### DATASET.INFO()
![image](https://github.com/user-attachments/assets/6eb9e699-e5e6-4f83-8f59-e387eb2b6737)
#### DATASET.DESCRIBE()
![image](https://github.com/user-attachments/assets/38303258-78bc-49e5-b083-e7d70f8c85ce)

### OUTPUT

#### Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/6a6f3843-d470-475c-9530-c6e0de294a23)

#### Test Data Root Mean Squared Error
![image](https://github.com/user-attachments/assets/89054004-5ebc-46ab-9f6d-15cdaaa76869)

#### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/925d3dea-d77d-463c-84ad-04a0f0782e66)

### RESULT
Thus a neural network regression model for the given dataset is developed and the prediction for the given input is obtained accurately.
