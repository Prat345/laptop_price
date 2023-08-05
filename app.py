import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
import random

model = pickle.load(open('lr_model.pkl','rb'))
df = pd.read_csv('laptop_clean.csv', encoding='latin-1')
with open('label_encoded.json') as json_file:
    labels = json.load(json_file)
st.title("Laptop Price Prediction")

feature = ['Company', 'TypeName', 'Inches', 'Cpu', 'Ram', 'Gpu', 'OpSys', 'Weight',
       'ScreenType', 'Resolution', 'TouchScreen', 'Cpu_Frequency',
       'MemoryType', 'MemorySize']

def selectbox_catcol(topic):
    select = st.selectbox(topic, sorted(df[topic].dropna().unique().astype(str)))
    select = int(labels[topic][select])
    return select

company = selectbox_catcol('Company')

type = selectbox_catcol('TypeName')

inches = st.slider('Inches',min_value=10.1, max_value=18.4, value=15.6, step=0.1)

cpu = selectbox_catcol('Cpu')

cpu_freq = st.slider('Cpu Frequency',min_value=0.9, max_value=3.6, value=2.5, step=0.1)

ram = st.selectbox('Ram', sorted(df['Ram'].unique()))

gpu = selectbox_catcol('Gpu')

opsys = selectbox_catcol('OpSys')

weight = selectbox_catcol('Weight')

screentype = selectbox_catcol('ScreenType')

resolution = selectbox_catcol('Resolution')

touchscreen = st.selectbox('TouchScreen',['Yes', 'No'])
if touchscreen == 'Yes':
    touchscreen = 1
else:
    touchscreen = 0

mem_type = selectbox_catcol('MemoryType')

mem_size = st.selectbox('MemorySize', sorted(df['MemorySize'].unique()))

currency = st.radio('',('US dollars', 'Baht'))
query = np.array([[company, type, inches,
                cpu, ram, gpu, opsys, weight,
                screentype, resolution, touchscreen,
                cpu_freq, mem_type, mem_size]])

# st.write(query)
if st.button('Predict Price'):
    price = model.predict(query)[0]
    if currency == 'Baht':
        price *= 30
        price = round(price,2)
        price = str(price) + ' Baht'
    else:
        price = round(price,2)
        price = f'$ {str(price)}'
    
    st.write(f'''
                Predicted Price 
                **{price}**
            ''')

