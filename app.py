import joblib
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction App", page_icon="🏠", layout="centered"
)


# Load the trained pipeline
@st.cache_resource
def load_model():
  # Replace with your actual model path if needed
  model_path = "my_house_price_prediction_pipeline.joblib"
  return joblib.load(model_path)


try:
  pipeline = load_model()
except Exception as e:
  st.error(
      f"Error loading the model pipeline. Make sure 'my_house_price_prediction_pipeline.joblib' is in the same directory. Details: {e}"
  )
  st.stop()

st.title("🏠 House Price Prediction App")
st.write(
    "Enter the property details below to get an estimated market price prediction."
)

st.markdown("---")

# Create a form for user inputs
with st.form("prediction_form"):
  st.subheader("Property Features")

  col1, col2 = st.columns(2)

  with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
    bathrooms = st.number_input(
        "Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5
    )
    sqft_living = st.number_input(
        "Living Area (sqft)", min_value=200, max_value=15000, value=2000, step=50
    )
    sqft_lot = st.number_input(
        "Lot Area (sqft)", min_value=200, max_value=500000, value=5000, step=100
    )
    floors = st.number_input(
        "Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5
    )
    waterfront = st.selectbox("Waterfront View", [0, 1])
    view = st.slider("View Quality (0-4)", min_value=0, max_value=4, value=0)
    condition = st.slider(
        "Condition (1-5)", min_value=1, max_value=5, value=3
    )

  with col2:
    sqft_above = st.number_input(
        "Sqft Above Ground", min_value=200, max_value=15000, value=1500, step=50
    )
    sqft_basement = st.number_input(
        "Sqft Basement", min_value=0, max_value=10000, value=0, step=50
    )
    yr_built = st.number_input(
        "Year Built", min_value=1800, max_value=2026, value=1995
    )
    yr_renovated = st.number_input(
        "Year Renovated", min_value=0, max_value=2026, value=0
    )

    # Text inputs for categorical features (adjust or change to dropdowns if you have standard lists)
    street = st.text_input("Street Address", value="123 Example St")
    city = st.text_input("City", value="Seattle")
    statezip = st.text_input("State & Zip", value="WA 98103")

  submit_button = st.form_submit_button(
      label="Predict House Price", use_container_width=True
  )

# Process prediction upon clicking the button
if submit_button:
  # Construct a DataFrame matching the training feature structure (16 columns)
  input_data = pd.DataFrame({
      "bedrooms": [bedrooms],
      "bathrooms": [bathrooms],
      "sqft_living": [sqft_living],
      "sqft_lot": [sqft_lot],
      "floors": [floors],
      "waterfront": [waterfront],
      "view": [view],
      "condition": [condition],
      "sqft_above": [sqft_above],
      "sqft_basement": [sqft_basement],
      "yr_built": [yr_built],
      "yr_renovated": [yr_renovated],
      "street": [street],
      "city": [city],
      "statezip": [statezip],
  })

  try:
    # Because your model is bundled with the ColumnTransformer pipeline,
    # it automatically handles scaling and one-hot encoding internally.
    prediction = pipeline.predict(input_data)
    predicted_price = prediction[0]

    st.success("Prediction Successful!")
    st.metric(
        label="Estimated House Price", value=f"${predicted_price:,.2f}"
    )

  except Exception as e:
    st.error(
        f"An error occurred during prediction. Please verify input formats."
        f" Details: {e}"
    )
