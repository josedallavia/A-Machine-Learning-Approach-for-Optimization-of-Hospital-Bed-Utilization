import sys
import os
from thesis_lib.forecast import *

def main(date):
    print("*************************************************************")
    print("############## Hospital Discharges Forecaster ###############")
    print("*************************************************************")
    print('')
    print("*************************************************************")
    print("********************************* by: @github.com/josedallavia")
    print('')
    print('For more information visit: https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability')
    print('')
    print('')
    forecast_discharges(date)

if __name__ == '__main__':

    os.chdir('/Users/josefinadallavia/Documents/MIM/Tesis/AML-hospital/')
    # Input parameters
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    day = int(sys.argv[3])
    main(datetime.date(year, month, day))