import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as tts
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv('diabetes.csv')

def histogram(dataframe):
    dataframe.hist()
    plt.show()


#preprocessing
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

#scale and standardise all values:
df_scaled = preprocessing.scale(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled['Outcome'] = df['Outcome']

df = df_scaled

#split data into training, testing and validation sets
x = df.loc[:, df.columns != 'Outcome']
y = df.loc[:,'Outcome' ]
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2)
x_train, x_val, y_train, y_val = tts(x_train, y_train, test_size=0.2)

#create model: 8 input nodes, 2 hidden layers, 1 output layer

model = Sequential()
# first hidden layer
model.add(Dense(32, activation='relu', input_dim=8)) #we signify the number of inputs taken in initially
model.add(Dense(16, activation='relu'))
#output layer
model.add(Dense(1, activation='sigmoid')) # sigmoid function used here to provide binary output

#complie the model
model.compile(optimizer= 'adam', loss= 'binary_crossentropy',metrics=['accuracy'] )

#train model
model.fit(x_train, y_train, epochs=200)
score = model.evaluate(x_test, y_test)
print("testing accuracy:", (score[1]*100))






