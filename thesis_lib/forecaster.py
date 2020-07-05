import datetime
import sys
from thesis_lib.modelling.data import *
from thesis_lib.modelling.model import *
import json
import os

def forecast_discharges(date, confidence_thresholds=confidence_dict):

    """
    Takes a date and returns predicted discharges with different confidence levels
    date: (datetime.date object, required)
    confidence_thresholds: (dict, optional)
    """

    #Initialize dict to store output
    output = {'confidence %': [],
              'lower bound forecast': []}

    #load data for the corresponding date
    data = Data().load('data/hospital_dataset')
    date_data = data.test.X[data.test.X['date'] == date]

    assert len(date_data) > 0, 'No data available for the specified date'

    #Load model
    with open("experiments/optimized_gdbt_model.pkl", "rb") as input_file:
        lgbm_model = pkl.load(input_file)

    #transformed data and get the predictions
    transformed_data = lgbm_model.pipeline.transform(date_data)
    predictions = lgbm_model.predict(transformed_data)

    #Load confidence thresholds to use:
    with open('confidence_dict.json', 'r') as json_file:
        confidence_dict = json.load(json_file)

    #Compute forecasts:
    for confidence_level in confidence_thresholds:
        threshold = confidence_thresholds[confidence_level]
        forecast = sum([1 for prob in predictions if prob > threshold])

        print(('With {p} % of confidence, number of discharges will be at least {N}'
               .format(p=confidence_level, N=forecast)))

        output['confidence %'].append(confidence_level)
        output['lower bound forecast'].append(forecast)

    return pd.DataFrame(output)

def main(date):
    forecast_discharges(date)

if __name__ == '__main__':

    os.chdir('/Users/josefinadallavia/Documents/MIM/Tesis/AML-hospital/')

    #Input parameters
    year = sys.argv[1]
    month = sys.argv[2]
    day = sys.argv[3]

    main(datetime.date(year,month,day))


