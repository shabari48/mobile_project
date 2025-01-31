import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import time

# Load your dataset, regression model, classification model, and scaler
@st.cache_data
def load_data():
    data = pd.read_csv("unique.csv")
    return data

@st.cache_data
def load_regression_model():
    model = joblib.load("myreg.joblib")
    return model

@st.cache_data
def load_classification_model():
    model = joblib.load("myclassify.joblib")
    return model

@st.cache_data
def load_mainscaler():
    scaler = joblib.load("mainscaler.joblib")
    return scaler

@st.cache_data
def load_kmeanscaler():
    scaler = joblib.load("kmeanscaler.joblib")
    return scaler

data = load_data()
regression_model = load_regression_model()
classification_model = load_classification_model()
mainscaler = load_mainscaler()
kmeanscaler = load_kmeanscaler()

# Define categories and their criteria
categories = {
    "Best for Gaming": {
        "criteria": ["Battery Power(in mAh)", "RAM", "Memory", "Processor Performance", "Screen Height", "Screen Width"],
        "weights": [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
    },
    "Best for Camera": {
        "criteria": ["Primary Camera (in MP)", "Front Camera (in MP)", "Screen Height", "Screen Width"],
        "weights": [0.4, 0.3, 0.2, 0.1],
    },
    "Best for Students": {
        "criteria": ["Battery Power(in mAh)", "Screen Height", "Screen Width", "Memory"],
        "weights": [0.3, 0.3, 0.2, 0.2],
    }
}

# Streamlit app layout
st.title("Phone Price Prediction and Recommendation System")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Price Prediction", "Phone Recommendations", "EMI Calculator"])


if page == "Price Prediction":

    st.header("Phone Price Prediction")
    st.write("Enter the features of the phone to predict its price.")

    # Input fields for features
    battery = st.number_input("Battery Power (in mAh)", min_value=533, max_value=7000, value=4000)
    ram = st.pills("RAM", ["3GB", "4GB", "6GB", "8GB", "12GB", "16GB", "24GB"], default="6GB")
    memory = st.pills("Memory", ["32GB", "64GB", "128GB", "256GB", "512GB", "1TB"], default="128GB")
    processor = st.number_input("Processor Performance", min_value=0.0, max_value=5.0, value=3.0)
    primary_camera = st.number_input("Primary Camera (in MP)", min_value=12, max_value=200, value=12)
    front_camera = st.number_input("Front Camera (in MP)", min_value=8, max_value=40, value=8)

    # Convert RAM and Memory selection to a number
    ram_mapping = {
        "3GB": 3,
        "4GB": 4,
        "6GB": 6,
        "8GB": 8,
        "12GB": 12,
        "16GB": 16,
        "24GB": 24
    }

    memory_mapping = {
        "32GB": 32,
        "64GB": 64,
        "128GB": 128,
        "256GB": 256,
        "512GB": 512,
        "1TB": 1024
    }

    n_memory = memory_mapping[memory]
    n_ram = ram_mapping[ram]

    # Predict button
    if st.button("Predict Price",icon="🔮"):
        
        progress_bar = st.progress(0)

        for i in range(100):
            progress_bar.progress(i + 1,text=f"We are choosing the best mobiles for you... {i + 1}%")
            time.sleep(0.02) 
            
        input_data = pd.DataFrame({
            "Battery Power(in mAh)": [battery],
            "RAM": [n_ram],
            "Memory": [n_memory],
            "Processor Performance": [processor],
            "Primary Camera (in MP)": [primary_camera],
            "Front Camera (in MP)": [front_camera]
        })

        # Standardize the input data
        input_data_scaled = mainscaler.transform(input_data)

        # Make predictions
        predicted_price = regression_model.predict(input_data_scaled)[0]
        predicted_category = classification_model.predict(input_data_scaled)[0]
        st.success(f"Predicted Price: ₹{predicted_price:.2f}")
        st.success(f"Predicted Category: {predicted_category}")

        # Include the predicted price in the input data for clustering
        input_data_with_price = input_data.copy()
        input_data_with_price['Price'] = predicted_price

        # Apply clustering algorithm to find similar phones
        num_clusters = 3
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)

        # Scale the data for clustering, including the price
        data_scaled = kmeanscaler.transform(data[["Battery Power(in mAh)", "RAM", "Memory", "Processor Performance", "Primary Camera (in MP)", "Front Camera (in MP)", "Price"]])
        data["Cluster"] = kmeans.fit_predict(data_scaled)

        # Find the cluster of the predicted phone
        input_data_with_price_scaled = kmeanscaler.transform(input_data_with_price)
        input_data_cluster = kmeans.predict(input_data_with_price_scaled)[0]

        # Filter phones in the same cluster
        similar_phones = data[data["Cluster"] == input_data_cluster]
        
        similar_phones_copy = similar_phones.copy()
        similar_phones_copy["Price_Difference"] = abs(similar_phones["Price"] - predicted_price)
    
        
        top_phones = similar_phones_copy.sort_values(by="Price_Difference",ascending=True).head(10)
        top_phones =top_phones.sort_values(by="Ratings",ascending=False)

        # Display the top phones
        st.write(f" <h3 style='color: #FF4B4B;'> These are the 10 Phones that may interest you </h3>", unsafe_allow_html=True)
        st.write(top_phones[["Brand", "Mobile Name", "Price", "Ratings", "RAM", "Memory", "Processor Performance"]])

elif page == "Phone Recommendations":
    st.header("We suggest you the best phones based on your preferences.")
    st.write("Select a category and price range to get top phone recommendations.")

    # Dropdown for category selection
    category = st.selectbox("Select Category", list(categories.keys()))

    # Slider for price range selection
    min_price, max_price = st.slider("Select Price Range", min_value=int(data["Price"].min()), max_value=int(data["Price"].max()), value=(int(data["Price"].min()), int(data["Price"].max())))

    # Filter data based on selected price range
    filtered_data = data[(data["Price"] >= min_price) & (data["Price"] <= max_price)]

    # Get criteria and weights for the selected category
    criteria = categories[category]["criteria"]
    weights = categories[category]["weights"]

    # Calculate weighted scores for each phone
    filtered_data_copy = filtered_data.copy()
    filtered_data_copy["Score"] = filtered_data_copy[criteria].dot(weights)

    # Sort the filtered data based on scores and ratings
    top_phones = filtered_data_copy.sort_values(by=["Score", "Ratings"], ascending=[False, False]).head(10)

    # Display top phones
    st.write(f" <h3 style='color: #FFFFFF;'> Top Rated Phones in Category  <span style='color: #FF4B4B;'>{category}</span> and Price Range  <span style='color: #FF4B4B;'>₹{min_price} - ₹{max_price}</span></h3>", unsafe_allow_html=True)

    st.write(top_phones[["Brand", "Mobile Name", "Price", "Ratings", "RAM", "Memory", "Processor Performance"]])
    
    
elif page == "EMI Calculator":
    st.header("EMI Calculator")
    st.write("Enter your details to calculate the EMI and the number of months required to pay off the phone.")

    # Input fields for EMI calculation
    salary = st.number_input("Monthly Salary (in ₹)", min_value=10000, max_value=1000000, value=50000)
    phone_price = st.number_input("Phone Price (in ₹)", min_value=5000, max_value=200000, value=50000)
    down_payment = st.number_input("Down Payment (in ₹)", min_value=0, max_value=200000, value=10000)
    interest_rate = st.number_input("Interest Rate (% per annum)", min_value=0.0, max_value=30.0, value=12.0)
    emi_tenure = st.number_input("EMI Tenure (in months)", min_value=1, max_value=36, value=12)

    # Calculate EMI
    if st.button("Calculate EMI"):
        loan_amount = phone_price - down_payment
        monthly_interest_rate = interest_rate / 12 / 100
        emi = (loan_amount * monthly_interest_rate * (1 + monthly_interest_rate) ** emi_tenure) / ((1 + monthly_interest_rate) ** emi_tenure - 1)
        st.success(f"Monthly EMI: ₹{emi:.2f}")
        st.success(f"Total Repayment Amount: ₹{(emi * emi_tenure):.2f}")
        st.success(f"Total Interest Paid: ₹{((emi * emi_tenure) - loan_amount):.2f}")