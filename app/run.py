import sys
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Scatter, Pie
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

database_name = sys.argv[1] if len(sys.argv) > 1 else "MelbourneHousing.db"
model_name = sys.argv[2] if len(sys.argv) > 2 else "predictor.pkl"
 
# load data
engine = create_engine(f'sqlite:///../data/{database_name}')
df = pd.read_sql_table('Data', engine)

# load model
model = joblib.load(f"../models/{model_name}")

# load column names
column_names = pd.read_csv("./model_column_names.csv").columns.to_list()

@app.route('/')
@app.route('/index')
def index():

    feature_importances = sorted(zip(model.named_steps['model'].feature_importances_, column_names), reverse=True)
    other_features = sum([x[0] for x in feature_importances[12:]])

    pie_values = list(map(lambda x: x[0],feature_importances[:12])) + [other_features]
    pie_labels = list(map(lambda x: x[1],feature_importances[:12])) + ["Other"]

    # create visuals
    graphs = [
        {
            'data': [
                Scatter(
                    x = df["Lattitude"].to_list(),
                    y = df["Longtitude"].to_list(),
                    text = ("AU$ " + df["Price"].astype(str)).to_list(),
                    mode = "markers",
                    marker={"color": df["Price"].to_list(), "opacity": 0.7}
                )
            ],

            'layout': {
                'title': 'Melbourne Housing Market',
                'width': 700,
                'height': 700,
            }
        },
        {
            'data': [
                Pie(
                    values= pie_values,
                    labels= pie_labels
                )
            ],

            'layout': {
                'title': 'Features that effect listing price',
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'GET':
        suburbs = sorted(list(df["Suburb"].unique()))

        return render_template('predict.html', suburbs=suburbs)
    elif request.method == 'POST':
        data = request.form 

        # Impute values we don't expect the user to know
        property_count = df.groupby("Suburb")[["Propertycount"]].mean().loc[data["suburb"]].Propertycount
        region_name = df[df["Suburb"] == data["suburb"]]["Regionname"].unique()[0]
        distance = df.groupby("Suburb")[["Distance"]].mean().loc[data["suburb"]].Distance

        # The training data rooms is based off the bedrooms so we don't let the user change this value
        rooms = data["bedrooms"]

        # Helper function to transform the property age
        def categorize_year(x):
            """
            Get the period in which the property was built from the YearBuilt column

            Args:
                X (int): Year

            Returns:
                string: Period of time
            """

            if x >= 2000:
                return "2000-2020"

            if x < 2000 and x >= 1980:
                return "1980-1999"

            if x < 1980 and x >= 1960:
                return "1960-1979"

            if x < 1960 and x >= 1940:
                return "1940-1959"

            return "before 1940"

        # Build property data
        property = {"Suburb": data["suburb"], "Rooms": int(rooms), "Type": data["type"], "Distance": distance,
                "Bedroom2": float(data["bedrooms"]), "Bathroom": float(data["bathrooms"]),
                "Car": float(data["car"]), "Landsize": float(data["landsize"]), "Lattitude": float(data["latitude"]), 
                "Longtitude": float(data["longitude"]), "Regionname": region_name, "Propertycount": property_count, 
                "PeriodBuilt": categorize_year(2018 - int(data["age"])) if data["age"] else None, "YearSold": 2018
                }

        property_df = pd.DataFrame({k: [v] for k, v in property.items()})

        result = model.predict(property_df)[0]

        return render_template('predict.html', result=result)


@app.route('/notebook')
def notebook():
    return render_template(
        'notebook.html',
    )

@app.route('/notebook/get')
def get_notebook():
    return render_template("MelbourneHousingPrice.html")


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()