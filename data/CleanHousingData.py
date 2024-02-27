import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CleanHousingData(BaseEstimator, TransformerMixin):
    # Map of Suburbs and their centre location
    suburb_location_map = {
            "Burwood": {"latitude": -37.84879, "longitude": 145.10812},
            "Clifton Hill": {"latitude": -37.78758, "longitude": 144.99984},
            "Footscray": {"latitude": -37.79862, "longitude": 144.89782},
            "Hampton East": {"latitude": -37.93836, "longitude": 145.03155},
            "Williamstown North": {"latitude": -37.85173, "longitude": 144.86235},
            "Melbourne": {"latitude": -37.81394, "longitude": 144.96572},
            "Brooklyn": {"latitude": -37.81746, "longitude": 144.84659},
            "North Melbourne": {"latitude": -37.80090, "longitude": 144.94888},
            "Oakleigh South": {"latitude": -37.92603, "longitude": 145.09528},
            "Essendon": {"latitude": -37.75286, "longitude": 144.91533},
            "Seddon": {"latitude": -37.80671, "longitude": 144.89225},
            "Croydon": {"latitude": -37.79465, "longitude": 145.28255},
            "Kensington": {"latitude": -37.79404, "longitude": 144.92778},
            "Strathmore": {"latitude": -37.73414, "longitude": 144.91948},
            "Keysborough": {"latitude": -37.99932, "longitude": 145.17356},
            "Lalor": {"latitude": -37.66561, "longitude": 145.01678},
            "Mickleham": {"latitude": -37.53447, "longitude": 144.90723},
            "Wollert": {"latitude": -37.60879, "longitude": 145.03201},
            "Greenvale": {"latitude": -37.63989, "longitude": 144.88238},
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        """
        Cleans the data by applying a series of transformations such as dropping duplicates, handling NaN values, refactoring string values where neccessary

        Args:
            X (DataFrame): The DataFrame to clean

        Returns:
            DataFrame: Cleaned DataFrame
        """

        # Drop duplicate rows, unneccessary columns, missing target values, and majority missing rows
        cleaned_X = X.drop_duplicates()
        cleaned_X = cleaned_X.drop(["Address", "Method", "SellerG", "Postcode", "CouncilArea"], axis=1)
        cleaned_X.dropna(subset=["Price"], inplace=True)
        cleaned_X.dropna(subset=["Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt"], how="all", inplace=True)

        # Fill missing Bedroom value with the room value
        mask = cleaned_X["Bedroom2"].isnull()
        cleaned_X.loc[mask, 'Bedroom2'] = cleaned_X['Rooms']

        # Fill missing Bathroom value with the average bathroom value of properties with the same number of Bedrooms
        bathroom_map = cleaned_X.groupby(by=["Bedroom2"])[["Bathroom"]].mean()
        def get_bathroom(x):
            """
            Fill missing Bathroom value with the average bathroom value of properties with the same number of Bedrooms as the input property.

            Args:
                X (Series): Property listing row

            Returns:
                Series: Property listing row with updated bathroom value
            """
            bedroom_num = x.Bedroom2
            x.Bathroom = round(bathroom_map.loc[bedroom_num].Bathroom)
            return x
    
        cleaned_X[cleaned_X["Bathroom"].isnull()] = cleaned_X[cleaned_X["Bathroom"].isnull()].apply(get_bathroom, axis=1)

        # Fill missing Car value with the average car value of properties with the same number of Bedrooms and Rooms
        car_map = cleaned_X.groupby(by=["Rooms", "Bedroom2"])[["Car"]].mean()
        def get_car(x):
            """
            Fill missing Car value with the average car value of properties with the same number of Bedrooms and Rooms as the input property.

            Args:
                X (Series): Property listing row

            Returns:
                Series: Property listing row with updated car value
            """
            room_num = x.Rooms
            bedroom_num = x.Bedroom2
            x.Car = round(car_map.loc[room_num, bedroom_num].Car)
            return x
        
        cleaned_X[cleaned_X["Car"].isnull()] = cleaned_X[cleaned_X["Car"].isnull()].apply(get_car, axis=1)

        # Fill missing location values with the center location of the suburb they are in
        def fill_lat_lon(x):
            """
            Fill missing longitude and latitude with the center location of the suburb the property belongs to.

            Args:
                X (Series): Property listing row

            Returns:
                Series: Property listing row with updated latitude and longitude
            """
            sub = self.suburb_location_map[x.Suburb]
            x.Lattitude = sub['latitude']
            x.Longtitude = sub['longitude']
            return x

        cleaned_X[(cleaned_X["Lattitude"].isnull()) & (cleaned_X["Longtitude"].isnull())] = cleaned_X[(cleaned_X["Lattitude"].isnull()) & (cleaned_X["Longtitude"].isnull())].apply(fill_lat_lon ,axis=1)

        # Fill the missing landsize values with the building area
        missing_landsize = (cleaned_X.Landsize.isnull()) | (cleaned_X.Landsize == 0)
        cleaned_X.loc[missing_landsize, "Landsize"] = cleaned_X['BuildingArea']
        cleaned_X.drop("BuildingArea", axis=1, inplace=True)

        # Impute the final missing landsize values by getting the average landsize for properties that share the same type and number of rooms
        landsize_map = cleaned_X.groupby(by=["Type", "Rooms"])[["Landsize"]].mean()

        def get_landsize(x):
            """
            Update the landsize value by getting the average landsize for properties that share the same type and number of rooms as the input property

            Args:
                X (Series): Property listing row

            Returns:
                Series: Property listing row with updated landsize
            """
            pType = x.Type
            rooms_num = x.Rooms
            if (np.isnan(landsize_map.loc[pType, rooms_num].Landsize)):
                x.Landsize = cleaned_X.Landsize.mean()
            else:
                x.Landsize = round(landsize_map.loc[pType, rooms_num].Landsize)
        
            return x
        
        cleaned_X[(cleaned_X.Landsize.isnull()) | (cleaned_X.Landsize == 0)] = cleaned_X[(cleaned_X.Landsize.isnull()) | (cleaned_X.Landsize == 0)].apply(get_landsize, axis=1)

        # Convert the YearBuilt column into a category of the period in which it was built
        def categorize_year(x):
            """
            Get the period in which the property was built from the YearBuilt column

            Args:
                X (int): Year

            Returns:
                string: Period of time
            """
            if np.isnan(x):
                return np.nan

            if x >= 2000:
                return "2000-2020"

            if x < 2000 and x >= 1980:
                return "1980-1999"

            if x < 1980 and x >= 1960:
                return "1960-1979"

            if x < 1960 and x >= 1940:
                return "1940-1959"

            return "before 1940"
        
        cleaned_X["PeriodBuilt"] =  cleaned_X.YearBuilt.apply(categorize_year)
        cleaned_X.drop("YearBuilt", axis=1, inplace=True)

        # Convert the date sold to the year it was sold in
        cleaned_X["YearSold"] = pd.to_numeric(cleaned_X.Date.str[-4:])
        cleaned_X.drop("Date", axis=1, inplace=True)

        return cleaned_X