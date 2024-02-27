# Melbourne Housing Prices

In this project we are going to explore the housing market for Melbourne from 2016 to 2018. This project aims to solve the problem of many residents who do not know what factors lead to a good market value for their property. At the end of this project, users will be able to use the information to make informed decisions when buying new property or selling their own property in Melbourne.

We will achieve this by creating a model that can accurately evaulate a properties market value based on its attributes.

I will be using the dataset provided on Kaggle (https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market) to understand the housing market in Melbourne. This dataset provides features such as: Number of rooms, price, suburb, number of bedrooms and bathrooms, location and more.

This project only uses Melbourne housing data but once the model has been trained and deemed sucessfull, it can be applied to any location provided they include the same attributes.

_Note: if this model works for Melbourne there is no guarentee that it will be useful for other markets due to location specific trends._

## Run Locally

Clone the project

```bash
git clone https://github.com/cameron-dryden/melbourne-housing-prices
```

Go to the project directory

```bash
cd melbourne-housing-prices
```

Create conda environment to install dependencies

```bash
conda create --name <env> --file requirements.txt
```

Activate the conda environment

```bash
conda activate <env>
```

Start the web server

_Note: Because the pretrained model is too large for github, it is not included in this repository. You will need to train the model before starting the web server._

```bash
cd app
python run.py
```

## Model Setup & Deployment

#### Clean and save csv data to database

```bash
cd data
python process_data.py melbourne_housing.csv MelbourneHousing.db
```

| Parameter           | Required | Description                                                                                                       |
| :------------------ | :------- | :---------------------------------------------------------------------------------------------------------------- |
| `CSV data filepath` | `True`   | Path to the melbourne housing csv file.                                                                           |
| `Database Name`     | `True`   | The name to save the database to. (Keep it the same as the example to make it easier when running the web server) |

#### Train the model and save it

```bash
cd models
python train_model.py ../data/MelbourneHousing.db predictor.pkl
```

| Parameter    | Required | Description                                                                                                    |
| :----------- | :------- | :------------------------------------------------------------------------------------------------------------- |
| `Database`   | `True`   | Path to the cleaned melbourne housing database.                                                                |
| `Model Name` | `True`   | The name to save the model to. (Keep it the same as the example to make it easier when running the web server) |

#### Run the web server

```bash
cd app
python run.py MelbourneHousing.db predictor.pkl
```

_Note: Both parameters are optional. If you cleaned and trained the model using the example file names, then you don't need these parameters and can just use_ `python run.py` _otherwise you must specify the names of the database and model you used._
| Parameter | Required | Description |
| :-------- | :------- | :------------------------- |
| `Database Name` | `False` | The file name of the cleaned melbourne housing database. |
| `Model Name` | `False` | The file name of the model. |

## Implementation

This project required data analysis and cleaning, machine learning model training and building a front-end webpage to interact with the model.

#### Data Analysis and Cleaning

_For more insight regarding the data analysis and cleaning phase, please refer to the Jupyter Notebook in this repository or by hosting the webpage and selecting the "Notebook" menu item._

#### Measuring Performance (Metrics)

In order to decide on the best performing model (and ultimately what factors effect a properties market value), I used the Root Mean Squared Error metric to measure performance.

The reason I chose this metric was the fact that it puts a larger weighting on big errors. This allows us to differentiate between small price prediction differences and larger price prediction differences. This differentiation is important as if the model predicts a property price that is very close to the actual price, it should not be an error as there will naturally be variance with property prices. However, if the predicted property price has a large gap from the actual price, we can be certain that this is an error and should significantly effect the model's score.

#### Model Training

_An in-depth analysis of the different machine learning models can be found in the Jupyter Notebook_

The final model uses a Random Forest Regressor to make price predictions. This model has been fine-tuned using cross-validation along with RMSE to find the best parameters.

The reason Random Forest Regressor was used (besides having the most promising performance) is because of the functionality to view the best features which is vital to answering our problem statement.

The average RMSE score for the final model is ~290000

#### Webpage

The webpage consists of 4 pages:

- **Home:** This page has an overview of the training data which is displayed as a geographical representation of all the listings. It shows listing density as well as the pricing distribution in Melbourne. It also shows the features that are consiidered by the model as the top factors for determing housing prices in Melbourne.

- **Price Predictions:** This page is the user-interface to interacting with the model. It offers form fields for users to enter in property features and in return gives the predicted price of the property in Australian dollars.

- **Notebook:** Here users can view the Jupyter Notebook where all the data analysis, cleaning and model exploration occured.

- **GitHub:** The link leads here

## Outcomes & Improvements

Overall this project was a success. I was able to make price predictions with relative accuaracy as well as being able to understand what factors effect a property's market value (These results can be found in the Notebook).

This project really encompasses the data science pipeline from begining to end. It involved extensive data cleaning and analysis, various machine learning models to find the best option and software development to build a user facing webpage.

There are some improvements that I have identified to improve the model performance. One being that the input data included some misrepresented values which seem to be incorrecly inputted (Some examples are Landsize values of 1, Bedroom number being greater than the room number and more). Looking at a more reliable source of input data could improve the model's performance.

Another improvement could be to bring the model up to speed on the latest property trends by training it on more recent data. The input data only has values up to 2018 which is quite outdated.

## File Structure

#### melbourne-housing-prices

| Path        | File                            | Description                                                                                                                                                                                                                                  |
| :---------- | :------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `./`        | `requirements.txt`              | Text file containing all the python packages used. (Can be used to create a conda environment will all the required packages)                                                                                                                |
|             | `Melbourne Housing Price.ipynb` | Jupyter Notebook with all the data analysis, cleaning and model exploration code.                                                                                                                                                            |
| `./data/`   | `process_data.py`               | Python file with ETL pipeline to clean the data. (Refer to Model Setup & Deployment for usage details)                                                                                                                                       |
|             | `melbourne_housing.csv`         | CSV file containing the raw data obtained from [Kaggle](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market)                                                                                                                |
|             | `MelbourneHousing.db`           | SQL Database containting the cleaned data exported from the process_data file                                                                                                                                                                |
|             | `CleanHousingData.py`           | Python helper file that contains all the code that is specific to cleaning the Melbourne housing data                                                                                                                                        |
| `./models/` | `train_model.py`                | Python file with ML pipeline that trains the model from the data and saves it. (Refer to Model Setup & Deployment for usage details)                                                                                                         |
|             | `predictor.pkl`                 | Pickle file containing the saved model. This is used in the web server to perform price predictions. _Note: Due to size of the model, it is not included in this GitHub repo and is required to be built if you want to run the web server._ |
| `./app/`    | `run.py`                        | Starts a web server which hosts the web interface to interact with the model and perform inference. (Refer to Model Setup & Deployment for usage details)                                                                                    |
|             | `model_column_names.csv`        | CSV File with the transformed column names which is used by the web server to display the top features.                                                                                                                                      |

## Acknowledgements

- [Melbournce Housing Market](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market)
- [Udacity Data Science Course](https://www.udacity.com/enrollment/nd025/5.0.5)
- [Leaflet Maps Library](https://leafletjs.com/)
- [Scikit-Learn Machine Learning](https://scikit-learn.org/stable/index.html)
- [Pandas Data Analysis Tool](https://pandas.pydata.org/docs/index.html)
- [Flask](https://flask.palletsprojects.com/en/3.0.x/)
