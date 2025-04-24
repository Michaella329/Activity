import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.title("🌸 Iris Flower Classification App")
st.write("This app uses **machine learning** to classify iris flowers into setosa, versicolor, or virginica.")

# Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Sidebar
st.sidebar.header("User Input Parameters")
sepal_length = st.sidebar.slider('Sepal length', float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()))
sepal_width = st.sidebar.slider('Sepal width', float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()))
petal_length = st.sidebar.slider('Petal length', float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()))
petal_width = st.sidebar.slider('Petal width', float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()))

# Model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df["target"])

prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
predicted_species = iris.target_names[prediction[0]]

st.write(f"### 🧠 Prediction: **{predicted_species}**")

# Show a plot
st.write("### 📊 Data Visualization")
fig = sns.pairplot(df, hue="target", corner=True)
st.pyplot(fig)
