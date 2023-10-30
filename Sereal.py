import pickle
import streamlit as st

model = pickle.load(open('Sereal.sav', 'rb'))

st.title('Estimasi Sereal')

calories = st.number_input('Masukan Calory')
protein = st.number_input('Protein')
fat = st.number_input(
    'fat')
sodium = st.number_input('Sodium')
fiber = st.number_input('fiber')
carbo = st.number_input('carbo')
sugars = st.number_input('sugars')
potass = st.number_input(
    'potas')
vitamins = st.number_input(
    'vitamins', step=0, max_value=250)
shelf = st.number_input('sehlf')
weight = st.number_input(
    'weight')
cups = st.number_input('cups')
rating = st.number_input('rating')

predict = ''

if st.button(' Estimasi Sereal'):
    predict = model.predict(
        [[calories, protein, fat,sodium, fiber, carbo, sugars, potass, vitamins, shelf, weight, cups, rating]]
    )
    st.write('Estimasi Sereal: ', predict)
    st.write('Estimasi Sereal: ', predict*2000)
