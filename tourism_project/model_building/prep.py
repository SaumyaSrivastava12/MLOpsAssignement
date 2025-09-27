# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("MLOps_token"))
DATASET_PATH = "hf://datasets/sammysri12/Tourism-Package-Prediction/tourism.csv"
bank_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',                        # Customer's age
    'NumberOfPersonVisiting',     # Total number of people accompanying the customer on the trip.
    'PreferredPropertyStar',      # Preferred hotel rating by the customer.
    'NumberOfTrips',              # Average number of trips the customer takes annually.
    'Passport',                   # Whether the customer holds a valid passport (0: No, 1: Yes).
    'OwnCar',                     # Whether the customer owns a car (0: No, 1: Yes).
    'NumberOfChildrenVisiting',   # Number of children below age 5 accompanying the customer.
    'MonthlyIncome',              # Gross monthly income of the customer.
    'PitchSatisfactionScore',     # Score indicating the customer's satisfaction with the sales pitch.
    'NumberOfFollowups',          # Total number of follow-ups by the salesperson after the sales pitch.-
    'DurationOfPitch',            # Duration of the sales pitch delivered to the customer
    ]

# List of categorical features in the dataset
categorical_features = [
    'CityTier',            # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).
    'TypeofContact',       # Company Invited or Self Inquiry
    'Occupation',          # Customer's occupation (e.g., Salaried, Freelancer).
    'Gender',              # Gender of the customer (Male, Female).
    'MaritalStatus',       # Marital status of the customer (Single, Married, Divorced).
    'Designation',         # Customer's designation in their current organization.
    'ProductPitched'       # The type of product pitched to the customer.
]


# Perform data cleaning
    # Drop CustomerID as it's not needed for training
    if 'CustomerID' in bank_dataset.columns:
        bank_dataset = bank_dataset.drop('CustomerID', axis=1)
        print("Dropped CustomerID column.")

    # Handle missing values (Example: drop rows with any missing values)
    bank_dataset.dropna(inplace=True)
    print(f"Shape after dropping missing values: {bank_dataset.shape}")


# Define predictor matrix (X) using selected numeric and categorical features
X = tour_dataset[numeric_features + categorical_features]

# Define target variable
y = tour_dataset[target]

# Split the cleaned dataset into training and testing sets, and save them locally.


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


# Upload the resulting train and test datasets back to the Hugging Face data space.
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="sammysri12/Tourism-Package-Prediction",
        repo_type="dataset",
    )
