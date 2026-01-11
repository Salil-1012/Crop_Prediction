import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,r2_score
import pickle
data = pd.read_csv("Crop_recommendation.csv")
X = data.drop("label",axis=1)
y = data["label"]
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,test_size = 0.2,random_state = 42)
model = RandomForestClassifier()
model.fit(X_train,y_train)
prediction = model.predict(X_test)
pickle.dump(model,open("model.pkl","wb"))