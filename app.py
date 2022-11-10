import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request

# ML Model prediction
data = pd.read_csv("static/RealEstateDataset.csv")

x=data.drop(['Y house price of unit area', 'X5 latitude', 'X6 longitude', 'X1 transaction date', 'No'], axis=1)
y=data['Y house price of unit area']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x, y)
accuracy = model.score(x_test, y_test)

# Flask application begins
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/estimate")
def estimate():
    name=request.args.get("name")
    age=request.args.get("age")
    metro=request.args.get("metro")
    store=request.args.get("store")
 
    new_data = pd.DataFrame({'X2 house age': [age], 
                    'X3 distance to the nearest MRT station': [metro], 
                    'X4 number of convenience stores': [store]})
    price=model.predict(new_data)

    return render_template("estimate-page.html", price=price[0], name=name, age=age, metro=metro, store=store, accuracy=accuracy)