
# Mobile Price Prediction and Recommendation System

## Overview

This project aims to predict the price of mobile phones based on their specifications and recommend similar phones using clustering techniques. The system uses a regression model to predict the price and a classification model to categorize the phones into different categories (Budget, Midrange, Flagship). Additionally, it employs KMeans clustering to find similar phones based on the predicted price and input data.

## Project Structure

```plaintext

mobile_project/
│
├── data/
│   ├── processed/
│   │   └── unique.csv
│   └── raw/
│
├── models/
│   ├── myreg.joblib
│   ├── myclassify.joblib
│   ├── mainscaler.joblib
│   └── kmeanscaler.joblib
│
├── notebooks/
│   ├── datatransformation.ipynb
│   ├── eda.ipynb
│   └── model.ipynb
│
├── src/
│   └── app.py
│
├── LICENSE
└── README.md

```

## Setup

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Streamlit
- Pandas
- Scikit-learn
- Joblib

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/mobile_project.git
   cd mobile_project
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Data

The dataset used in this project is stored in the `data/processed/` directory. The dataset contains various features of mobile phones, including battery power, RAM, memory, processor performance, camera specifications, and price.

## Models

The project uses the following models:

- **Regression Model**: Trained to predict the price of mobile phones based on their specifications.
- **Classification Model**: Trained to categorize mobile phones into different categories (Budget, Midrange, Flagship).
- **KMeans Clustering**: Used to find similar phones based on the predicted price and input data.

The models and scalers are saved in the `models/` directory.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for data transformation, exploratory data analysis (EDA), and model training.

- **datatransformation.ipynb**: Contains the code for data cleaning and transformation.
- **eda.ipynb**: Contains the code for exploratory data analysis.
- **model.ipynb**: Contains the code for training the regression and classification models.

## Streamlit App

The `src/app.py` file contains the code for the Streamlit application. The app allows users to input the specifications of a mobile phone and predict its price. It also recommends similar phones based on the predicted price and input data.

### Running the Streamlit App

1. **Activate the Virtual Environment**

   ```bash
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Run the Notebook File**

     ```bash
   model.ipynb
   ```

3. **Run the Streamlit App**

   ```bash
   streamlit run src/app.py
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

For any questions or suggestions, please feel free to contact us at [shabariprakashsv@gmail.com](mailto:shabariprakashsv@gmail.com).
