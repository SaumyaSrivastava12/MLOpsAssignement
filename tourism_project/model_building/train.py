# for data manipulation
import pandas as pd
# for creating a folder
import os
# for hugging face space authentication to upload files and download models
from huggingface_hub import login, HfApi, upload_file
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# for model training
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
# for model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# for experimentation tracking
import mlflow
import mlflow.sklearn

# Define constants
HF_DATASET_REPO_ID = "sammysri12/Tourism-Package-Prediction"
# Define a new repo for the model
HF_MODEL_REPO_ID = "sammysri12/Tourism-Package-Prediction"
# Assuming MLflow server is running locally
MLFLOW_TRACKING_URI = "http://localhost:5000"
# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load data from Hugging Face Hub
# Using the MLOps_token for dataset access
api = HfApi(token=os.getenv("MLOps_token"))
Xtrain = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO_ID}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO_ID}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO_ID}/ytrain.csv").squeeze() # Use squeeze to convert DataFrame to Series
ytest = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO_ID}/ytest.csv").squeeze()


print("Train and test datasets loaded successfully.")


# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()) # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models and their parameters for tuning
models = {
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5, 10]
        }
    },
    'Bagging': {
        'model': BaggingClassifier(random_state=42),
        'params': {
            'model__n_estimators': [10, 50, 100]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, 15]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1]
        }
    }
}

best_model = None
best_score = 0
best_model_name = ""

# Start MLflow run
with mlflow.start_run():
    for model_name, model_info in models.items():
        print(f"Training {model_name}...")
        # Create a pipeline with preprocessing and the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model_info['model'])])

        # Tune the model with the defined parameters
        grid_search = GridSearchCV(pipeline, model_info['params'], cv=3, scoring='f1')
        grid_search.fit(Xtrain, ytrain)

        # Log parameters and best score
        mlflow.log_param(f'{model_name}_best_params', grid_search.best_params_)
        mlflow.log_metric(f'{model_name}_best_f1_score', grid_search.best_score_)

        print(f"{model_name} best parameters: {grid_search.best_params_}")
        print(f"{model_name} best F1 score: {grid_search.best_score_}")

        # Evaluate the model performance on the test set
        y_pred = grid_search.best_estimator_.predict(Xtest)
        accuracy = accuracy_score(ytest, y_pred)
        precision = precision_score(ytest, y_pred)
        recall = recall_score(ytest, y_pred)
        f1 = f1_score(ytest, y_pred)
        roc_auc = roc_auc_score(ytest, y_pred)

        # Log test set metrics
        mlflow.log_metric(f'{model_name}_test_accuracy', accuracy)
        mlflow.log_metric(f'{model_name}_test_precision', precision)
        mlflow.log_metric(f'{model_name}_test_recall', recall)
        mlflow.log_metric(f'{model_name}_test_f1', f1)
        mlflow.log_metric(f'{model_name}_test_roc_auc', roc_auc)

        print(f"{model_name} Test Set Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")

        # Check if this is the best model so far based on F1 score on the test set
        if f1 > best_score:
            best_score = f1
            best_model = grid_search.best_estimator_
            best_model_name = model_name

    # Log the best model
    if best_model:
        print(f"\nBest model is: {best_model_name} with F1 score: {best_score:.4f}")
        mlflow.sklearn.log_model(best_model, "best_model")

        # Register the best model in the Hugging Face model hub
        HF_MODEL_REPO_ID = "sammysri12/Tourism_Package_Prediction_Model"
        # Create model repo if it doesn't exist
        try:
            api.repo_info(repo_id=HF_MODEL_REPO_ID, repo_type="model")
            print(f"Model Space '{HF_MODEL_REPO_ID}' already exists. Using it.")
        except RepositoryNotFoundError:
            print(f"Model Space '{HF_MODEL_REPO_ID}' not found. Creating new space...")
            api.create_repo(repo_id=HF_MODEL_REPO_ID, repo_type="model", private=False)
            print(f"Model Space '{HF_MODEL_REPO_ID}' created.")

        # Save the best model locally
        model_path = "best_model.pkl"
        import joblib
        joblib.dump(best_model, model_path)

        # Upload the model file to the Hugging Face model hub
        upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_model.pkl",
            repo_id=HF_MODEL_REPO_ID,
            repo_type="model",
            token=os.getenv("MLOps_token")
        )
        print(f"Best model '{best_model_name}' uploaded to Hugging Face model hub.")
