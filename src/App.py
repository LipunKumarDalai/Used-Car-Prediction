import streamlit as st 
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
import json
UsedCarModel = joblib.load(r"models/model.pkl")
oe = joblib.load(r"models/oe.pkl")
with open(r"data/json/brands","r") as fd:
    brand = json.load(fd)
fd.close()
with open(r"data/json/cars","r") as f:
    cars = json.load(f)
f.close()

st.header("UsedCar Price Model",divider=True)
st.text("A Simple App that will predict the Prices of UsedCars with 93% Accuracy.")
option = [i for i in brand.keys()]
st.subheader("Input Section")
brands = st.selectbox(label="Select Car's Brand(If possible)",options=option+["other"])
if brands=="other":
    brand = st.text_input(label="Enter Brand")
    car = st.text_input(label="Enter Car Name(e.g Maruti(brand) Suzuki(Model))")
    if len(car.split(" "))>1:
        Model = car.split(" ")[1]
    else:
        pass
else:
    car = st.selectbox(label="Select Car Name(If Possible)",options=cars[brands])
    Model = car.split(" ")[1]
vehicle_age = st.slider(label="Select Car's Age",min_value=0,max_value=30,step=1,value=5)
driven = st.number_input(label="Km driven",min_value=0,max_value=3000000,step=1,value=1000)
col1, col2, col3 = st.columns(3)
with col1:
    seller = st.radio("Select Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
with col2:
    Fuel = st.radio("Select Fuel Type", ["CNG", "Diesel", "Electric", "LPG", "Petrol"])
with col3:
    Transmission = st.radio("Select Transmission Type", ["Manual", "Automatic"])
mileage = st.number_input(label="Enter Mileage",max_value=40.0,value=10.1)
engine = st.number_input(label="Enter Engine",max_value=8500,min_value=550,step=1)
max_power = st.number_input(label="Enter HorsePower",max_value=700,min_value=30)
seats = st.selectbox(label="Select Seats",options=[2,3,4,5,6,7,8,9])

if st.button("submit"):
    cat = oe.transform([[car,brands,Model,seller,Fuel,Transmission]])
    cat = cat.tolist()
    cat1 = cat[0]
    num = [vehicle_age,driven,mileage,engine,max_power,seats]
    final = cat1 + num
    output = UsedCarModel.predict([final])
    st.success("Price Predicted")
    st.write("Predicted Price: Rs",abs(int(output[0])))




