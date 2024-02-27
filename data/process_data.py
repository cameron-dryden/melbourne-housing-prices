import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from CleanHousingData import CleanHousingData


def load_data(filepath):
    """
    Loads the provided dataset into a DataFrame

    Args:
        filepath (str): File path to the dataset
    
    Returns:
        DataFrame: DataFrame containing dataset
    """

    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame):
    """
    Cleans the data and rebuilds the cleaned data as a DataFrame

    Args:
        df (DataFrame): Input DataFrame

    Returns:
        DataFrame: Cleaned DataFrame
    """
    cleaner = CleanHousingData()
    cleaned_data = cleaner.fit_transform(df)

    return cleaned_data


def save_data(df, database_filename):
    """
    Save the data into a SQL database.

    Args:
        df (DataFrame): DataFrame to be saved
        database_filename (string): Database name to be saved to
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("Data", engine, index=False, if_exists="replace") 


def main():
    if len(sys.argv) == 3:

        data_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    DATA: {}'
              .format(data_filepath))
        df = load_data(data_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepath of the housing dataset and the name of the database'\
              '\n\nExample: python process_data.py melbourne_housing.csv MelbourneHousing.db')


if __name__ == '__main__':
    main()