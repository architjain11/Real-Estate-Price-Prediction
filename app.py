import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math


data = pd.read_csv("static/RealEstateDataset.csv")

x=data.drop(['Y house price of unit area', 'X5 latitude', 'X6 longitude', 'X1 transaction date', 'No'], axis=1)
y=data['X4 number of convenience stores']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)
y_pred = model.predict(x_test)
op = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(op)