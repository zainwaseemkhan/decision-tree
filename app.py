import streamlit as st
import joblib
import numpy as np

# Model load karen jo aapne Kaggle par save kiya tha
model = joblib.load('decision_tree_iris_model.pkl')

st.title('Iris Flower Species Prediction App')
st.write('Decision Tree Model ka istemal kar ke phool ki qisam (species) maloom karen.')

# User se input lene ke liye sliders
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.4)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.4)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.3)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

# Predict button
if st.button('Predict'):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    
    # Iris classes ke naam
    species = ['Setosa', 'Versicolor', 'Virginica']
    result = species[prediction[0]]
    
    st.success(f'Yeh phool **{result}** category ka hai!')