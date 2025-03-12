

# ANN for Regression: Customer Salary Estimation

## Overview

This project implements an Artificial Neural Network (ANN) to predict customer salaries based on various features. The model is developed using PyTorch and provides a Streamlit application for interactive user engagement.

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used is `Churn_Modelling.csv`, which includes the following features:

- **RowNumber**: Index of the row.
- **CustomerId**: Unique identifier for each customer.
- **Surname**: Customer's surname.
- **CreditScore**: Customer's credit score.
- **Geography**: Customer's country.
- **Gender**: Customer's gender.
- **Age**: Customer's age.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Customer's account balance.
- **NumOfProducts**: Number of products the customer has with the bank.
- **HasCrCard**: Indicates if the customer has a credit card (1: Yes, 0: No).
- **IsActiveMember**: Indicates if the customer is an active member (1: Yes, 0: No).
- **EstimatedSalary**: Customer's estimated salary.

**Note**: The target variable for this regression task is `EstimatedSalary`.

## Model Architecture

The ANN is constructed using PyTorch and consists of:

- **Input Layer**: Accepts the input features.
- **Hidden Layers**: Two hidden layers with ReLU activation functions.
- **Output Layer**: Outputs the estimated salary.

## Installation

To set up the environment, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/akurapati1/ANN-for-Regression-Customer-salary-Estimation-.git
   cd ANN-for-Regression-Customer-salary-Estimation-
   ```

2. **Install the required packages**:

   Ensure you have Python installed. Then, install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:

   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `torch`
   - `streamlit`

## Usage

1. **Training the Model**:

   The `salaryRegression.ipynb` notebook contains the code for preprocessing the data, building the ANN model, and training it. You can run this notebook to train the model on the dataset.

2. **Running the Streamlit Application**:

   After training the model, you can launch the Streamlit app to interact with the model:

   ```bash
   streamlit run streamlit_regression.py
   ```

   This will open a web interface where you can input customer features and get the estimated salary.

## Results

The model's performance is evaluated using Mean Squared Error (MSE) and R-squared metrics. Detailed results and visualizations are available in the `salaryRegression.ipynb` notebook.


## License

This project is licensed under the MIT License. 
