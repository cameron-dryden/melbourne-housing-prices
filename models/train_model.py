import csv
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine
import pickle


def load_data(database_filepath):
    """
    Loads the database from the specified path and splits it into the feature and target variables.

    Args:
        database_filepath (str): File path to the database.
    
    Returns:
        DataFrame: Feature columns.
        Series: Target column (price)
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("Data", engine)
 
    X = df.drop("Price", axis=1)
    y = df["Price"]

    return X, y



def build_model():
    """
    Builds a machine learning pipeline that will transform the data and fit a Random Forest model.

    Returns:
        Pipeline: Pipeline ready to fit
    """
    num_cols = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'Lattitude', 'Longtitude', 'Propertycount', 'YearSold' ]
    cat_cols = ['Suburb',  'Type', 'Regionname', 'PeriodBuilt']

    # Create transformation pipeline
    transformation_pipeline = Pipeline([
        ("columns", ColumnTransformer([
            ("numerical", StandardScaler(), num_cols),
            ("categorical", OneHotEncoder(), cat_cols)
        ])),
        ("model", RandomForestRegressor(max_features=0.8, n_estimators=100))
        ])

    return transformation_pipeline


def evaluate_model(model, X_test, y_test):
    """
    Provides performance results for the model using RMSE score.

    Args:
        model (RandomForestRegressor): Model to evaluate
        X_test (DataFrame): Features from the test dataset
        y_test (Series): Target from the test dataset
    """

    preds = model.predict(X_test) 

    print(f"RMSE score: {np.sqrt(mean_squared_error(y_test, preds))}")


def save_model(model, model_filepath):
    """
    Saves the model to the file system as a pkl file

    Args:
        model (RandomForestRegressor): Model to save
        model_filepath (str): Path to save the model to
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        
        # No need for train test split as the correct parameters have been found in the notebook.
        # We build the model on all the data to ensure that all Suburbs are included for One Hot Encoding

        print('Training model...')
        model = build_model()
        model.fit(X, y)

        print('Evaluating model...')
        evaluate_model(model, X, y)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the melbourne housing database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/MelbourneHousing.db predictor.pkl')


if __name__ == '__main__':
    main()