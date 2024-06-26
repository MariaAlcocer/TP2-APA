
import streamlit as st
import numpy as np

import pickle
from surprise import SVD, Dataset

# Cargar el modelo entrenado
with open('svd_model.pkl', 'rb') as model_file:
    svd_model = pickle.load(model_file)

st.title("Recomendador de Películas")

user_id = st.number_input("ID del Usuario", min_value=1)
item_id = st.number_input("ID del Ítem (Película)", min_value=1)

if st.button("Predecir"):
    prediction = svd_model.predict(user_id, item_id)
    st.write(f"La predicción para el usuario {user_id} y el ítem {item_id} es: {prediction.est}")
