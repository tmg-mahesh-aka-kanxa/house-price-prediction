import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load('my_house_price_prediction_pipeline.joblib')

model = load_model()

st.title("🏡 King County House Price Prediction")
st.write("Enter the property characteristics below to get a real-time price prediction.")

# Input widgets
col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input("Bedrooms", value=3, step=1)
    bathrooms = st.number_input("Bathrooms", value=2.0, step=0.5)
    sqft_living = st.number_input("Sqft Living", value=1800)
    sqft_lot = st.number_input("Sqft Lot", value=5000)
    floors = st.number_input("Floors", value=1.0, step=0.5)
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.slider("View Rating (0-4)", 0, 4, 0)

with col2:
    condition = st.slider("Condition (1-5)", 1, 5, 3)
    sqft_above = st.number_input("Sqft Above", value=1800)
    sqft_basement = st.number_input("Sqft Basement", value=0)
    yr_built = st.number_input("Year Built", value=1990)
    yr_renovated = st.number_input("Year Renovated", value=0)
    street = st.text_input("Street Address", value="123 Example St")
    city = st.text_input("City", value="Seattle")
    statezip = st.text_input("State / Zip", value="WA 98178")

if st.button("Predict Price", type="primary"):
    input_data = pd.DataFrame([{
        'bedrooms': float(bedrooms),
        'bathrooms': float(bathrooms),
        'sqft_living': int(sqft_living),
        'sqft_lot': int(sqft_lot),
        'floors': float(floors),
        'waterfront': int(waterfront),
        'view': int(view),
        'condition': int(condition),
        'sqft_above': int(sqft_above),
        'sqft_basement': int(sqft_basement),
        'yr_built': int(yr_built),
        'yr_renovated': int(yr_renovated),
        'street': str(street),
        'city': str(city),
        'statezip': str(statezip)
    }])
    
    prediction = model.predict(input_data)[0]
    st.success(f"### Estimated House Price: **${prediction:,.2f}**")
