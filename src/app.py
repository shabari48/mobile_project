import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load  dataset, regression model, classification model, and scaler
@st.cache_data
def load_data():
    data = pd.read_csv("../data/processed/unique.csv")
    return data

@st.cache_data
def load_regression_model():
    model = joblib.load("../models/myreg.joblib")
    return model

@st.cache_data
def load_classification_model():
    model = joblib.load("../models/myclassify.joblib")
    return model

@st.cache_data
def load_mainscaler():
    scaler = joblib.load("../models/mainscaler.joblib")
    return scaler

@st.cache_data
def load_kmeanscaler():
    scaler = joblib.load("../models/kmeanscaler.joblib")
    return scaler

@st.cache_data
def load_shasha():
    model=joblib.load("../models/shashaclassify.joblib")
    return model
    

data = load_data()
regression_model = load_regression_model()
classification_model = load_classification_model()
shasha_model=load_shasha()
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
page = st.sidebar.radio("Go to", ["Price Prediction", "Phone Recommendations","Phone Feature Comparison","EMI Calculator"])


if page == "Price Prediction":

    st.header("Phone Price Prediction")
    st.write("Enter the features of the phone to predict its price.")

    # Input fields for features
    battery = st.number_input("Battery Power (in mAh)", min_value=500, max_value=2000, value=1000)
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
        
        #Progress Bar
        progress_bar = st.progress(0)

        for i in range(100):
            progress_bar.progress(i + 1,text=f"We are picking the best mobiles for you... {i + 1}%")
            time.sleep(0.02) 
            
        
        # Input data for prediction
            
        input_data = pd.DataFrame({
            "Battery Power(in mAh)": [battery],
            "RAM": [n_ram],
            "Memory": [n_memory],
            "Processor Performance": [processor],
            "Primary Camera (in MP)": [primary_camera],
            "Front Camera (in MP)": [front_camera]
        })

        input_data_scaled = mainscaler.transform(input_data)

        # Predict price and category
        predicted_price = regression_model.predict(input_data_scaled)[0]
        predicted_category = classification_model.predict(input_data_scaled)[0]
        predicted_category_shasha=shasha_model.predict(input_data_scaled)[0]
        
      
        # Display the predicted price and category
        
        class_map={
            0:"Flagship",
            1:"Mid Range",
            2:"Budget"
        }
        
        st.success(f"Predicted Price: ₹{predicted_price:.2f}")
        st.success(f"Predicted Category by Sklearn: {class_map[predicted_category]}")
        st.success(f"Predicted Category by Shasha: {class_map[predicted_category_shasha]}")
        
        # Find similar phones using clustering
        
        input_data_with_price = input_data.copy()
        input_data_with_price['Price'] = predicted_price

        
        num_clusters = 4
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)

        # Fit KMeans clustering model
        
        data_scaled = kmeanscaler.transform(data[["Battery Power(in mAh)", "RAM", "Memory", "Processor Performance", "Primary Camera (in MP)", "Front Camera (in MP)", "Price"]])
        data["Cluster"] = kmeans.fit_predict(data_scaled)

       # Predict cluster for input data
       
        input_data_with_price_scaled = kmeanscaler.transform(input_data_with_price)
        input_data_cluster = kmeans.predict(input_data_with_price_scaled)[0]
        
        
        # Filter data based on cluster

        similar_phones = data[data["Cluster"] == input_data_cluster]
        
        
        #Pick top 10 phones based on price difference and ratings
        similar_phones_copy = similar_phones.copy()
        similar_phones_copy["Price_Difference"] = abs(similar_phones["Price"] - predicted_price)
    
        
        top_phones = similar_phones_copy.sort_values(by="Price_Difference",ascending=True).head(10)
        top_phones =top_phones.sort_values(by="Ratings",ascending=False)

        # Display the top phones
        st.write(f" <h3 style='color: #FF4B4B;'> These are the top 10 Phones that might interest you </h3>", unsafe_allow_html=True)
        st.write(top_phones[["Brand", "Mobile Name", "Price", "Ratings", "RAM", "Memory", "Processor Performance"]])
        
        

elif page == "Phone Recommendations":
    st.header("We suggest you the best phones based on your preferences.")
    st.write("Select a category, brand, and price range to get top phone recommendations.")

    # Dropdown for category selection
    category = st.selectbox("Select Category", list(categories.keys()))

    # Create an example phone with desirable features based on the selected category
    category_info = categories[category]
    category_criteria = category_info["criteria"]
    category_weights = category_info["weights"]
    
    example_phone = {}
    for criterion, weight in zip(category_criteria, category_weights):
        example_phone[criterion] = data[criterion].mean() * weight

    # Convert example phone to DataFrame
    example_phone_df = pd.DataFrame([example_phone])

    # Dropdown for brand selection
    unique_brands = data["Brand"].unique().tolist()  # Get unique brands from the dataset
    unique_brands.insert(0, "All Brands")  # Add "All Brands" as the default option
    selected_brand = st.selectbox("Select Brand", unique_brands)

    # Slider for price range selection
    min_price, max_price = st.slider("Select Price Range", min_value=int(data["Price"].min()), max_value=int(data["Price"].max()), value=(int(data["Price"].min()), int(data["Price"].max())))

    # Filter data based on selected price range
    filtered_data = data[(data["Price"] >= min_price) & (data["Price"] <= max_price)]

    # Further filter data based on selected brand (if not "All Brands")
    if selected_brand != "All Brands":
        filtered_data = filtered_data[filtered_data["Brand"] == selected_brand]

    # Calculate cosine similarity between example phone and filtered data
    scaler = StandardScaler()
    filtered_data_scaled = scaler.fit_transform(filtered_data[category_criteria])
    example_phone_scaled = scaler.transform(example_phone_df[category_criteria])
    similarity_scores = cosine_similarity(example_phone_scaled, filtered_data_scaled)[0]

    # Add similarity scores to filtered data
    filtered_data['Similarity_Score'] = similarity_scores

    # Sort by similarity score and ratings
    top_phones = filtered_data.sort_values(by='Similarity_Score', ascending=False).head(10)
    top_phones= top_phones.sort_values(by='Ratings',ascending=False)

    # Display top phones for selected category
    st.write(f"\n### Top Recommendations for {category}")
    features_to_display = ["Brand", "Mobile Name", "Price", "Ratings"]
    if category == "Best for Gaming":
        features_to_display.extend(["RAM", "Memory", "Processor Performance"])
    elif category == "Best for Camera":
        features_to_display.extend(["Primary Camera (in MP)", "Front Camera (in MP)"])
    elif category == "Best for Students":
        features_to_display.extend(["Battery Power(in mAh)", "Memory"])

    st.write(top_phones[features_to_display].to_html(index=False), unsafe_allow_html=True)
   


elif page == "Phone Feature Comparison":
    st.header("Phone Feature Comparison Visualizations")
    st.write("Select two phones and compare their features dynamically.")

    # Dropdowns for phone selection
    phone1 = st.selectbox("Select First Phone", data["Mobile Name"].unique())
    phone2 = st.selectbox("Select Second Phone", data["Mobile Name"].unique(), index=1)

    # Dynamic feature selection
    features = ["Battery Power(in mAh)", "RAM", "Memory", "Processor Performance",
                 "Primary Camera (in MP)", "Front Camera (in MP)"]
    selected_features = st.multiselect("Select Features to Compare", features, default=features[:3])

    if phone1 == phone2:
        st.warning("Please select two different phones for comparison.")
    elif selected_features:
        # Extract selected phone data
        phone1_data = data[data["Mobile Name"] == phone1][selected_features].iloc[0]
        phone2_data = data[data["Mobile Name"] == phone2][selected_features].iloc[0]

        # Plot bar graph for comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        index = range(len(selected_features))
        bar_width = 0.35

        # Create the bars
        ax.bar(index, phone1_data, bar_width, label=phone1, color="#3B6790")
        ax.bar([i + bar_width for i in index], phone2_data, bar_width, label=phone2, color="#EFB036")

        # Add value labels on top of bars
        for i, v in enumerate(phone1_data):
            ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
        for i, v in enumerate(phone2_data):
            ax.text(i + bar_width, v, f'{v:,.0f}', ha='center', va='bottom')

        ax.set_xlabel("Features")
        ax.set_ylabel("Values")
        ax.set_title("Phone Feature Comparison")
        ax.set_xticks([i + bar_width / 2 for i in index])
        ax.set_xticklabels(selected_features, rotation=45)
        ax.legend()

        st.pyplot(fig)
        
    else:
        st.warning("Please select at least one feature to compare.")
 
    
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