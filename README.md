# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: GURUMOORTHI R
### Register Number: 212222230042
```python

## Importing Required packages
```python
from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default

```
## Authenticate the Google sheet
```py
auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)
worksheet = gc.open('experiment1').sheet1
data = worksheet.get_all_values()
```
## Construct Data frame using Rows and columns
```py
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})
df.head()
x=df[['input']].values
y=df[['output']].values
x
```
## Split the testing and training data
```py
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1=Scaler.transform(X_train)
```
## Build the Deep learning Model
```py
network=Sequential([
    Dense(units=9,input_shape=[1]),
    Dense(units=9),
    Dense(units=1)
])
network.summary()
network.compile(optimizer = 'rmsprop', loss = 'mse')
network.fit(X_train1,y_train,epochs=1000)

loss_df = pd.DataFrame(network.history.history)
loss_df.plot()
```

## Evaluate the Model
```py
X_test1=Scaler.transform(X_test)
network.evaluate(X_test1,y_test)
X_1=[[5]]
X_1_1 = Scaler.transform(X_1)
network.predict(X_1_1)


```
## Dataset Information

![1](https://github.com/user-attachments/assets/d034a410-fc5e-4d2e-a09b-ad2e85efdeb7)


## OUTPUT

### Training Loss Vs Iteration Plot


![3](https://github.com/user-attachments/assets/7b259f6f-3fae-4275-9cf7-66f114ea198c)

### Test Data Root Mean Squared Error

![2](https://github.com/user-attachments/assets/61bccd2f-3705-4e1c-9f40-6e6812931b24)

![4](https://github.com/user-attachments/assets/b3d3d742-a8e7-4295-a824-c451ad4348d2)

### New Sample Data Prediction

![5](https://github.com/user-attachments/assets/3cd08070-475f-4ea6-b304-bb648467ae70)


## RESULT

Thus a Neural network for Regression model is Implemented Successfully
