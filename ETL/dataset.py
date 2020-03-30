import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('thesis_lib'))))
import thesis_lib
import pandas as pd
import shutil

import pandas as pd
import os
from os import path
import shutil
import matplotlib.pyplot as plt 
from random import random
import numpy as np 

from etl_helpers.data_processing import *
from thesis_lib.hospital_classes import *

from datetime import timedelta
from datetime import datetime

from collections import namedtuple

import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

def main(directory,origin_directory,prop=1):
    
    if path.exists(directory):
        print('Oops. The specified directory already exists')
    else:
        try:
            os.mkdir(directory)
        except:
            ValueError
    
        #Get patient clinic records
        docs = get_clinic_histories(origin_directory+'/patients_clinic_histories', prop)
        
        #Saved temporarily 
        docs.to_parquet(directory+'/patients_records_tmp.parquet')
       
        #Add patient static variables to dataset
        hospitalizations = load_parquet(origin_directory+'/parquet_data', 'hospitalizations')
        
        print('Adding patient static variables')
        patient_static_vars = ['birth_date','gender']
                                           
        df = add_static_variables(docs, 'patient_id', patient_static_vars, hospitalizations)
    
        #Add admission static variables to dataset
        print('Adding admission static variables')
        adm_static_vars = ['insurance_entity',
                        'entity_group',
                        'origin',
                        'admission_sector',
                        'isolation',
                        'administrative_diagnosis',
                        'presumptive_dianogsis',
                        'responsible_sector',
                        'emergency_service',
                        'new_born_weight',
                        'new_born_gestation_age',
                        'PIM2TEP',
                        'high_risk_TEP',
                        'low_risk_TEP',
                        'ARM_TEP',
                        'CEC_TEP',
                        'request_origin',
                        'request',
                        'request_diagnosis',
                        'request_sector',
                        'admission_date']
                        
        df = add_static_variables(df, 'admission_id', adm_static_vars , hospitalizations)
    
        #Compute uptodate patient age for each row
        df['patient_age'] = (df.date-df['birth_date']).astype(
                                                    'timedelta64[Y]').fillna(-1).astype('int')
        #Sort dataset                   
        df = df.sort_values(by='date',ascending=False)
    
        #Save dataset in the specified directory
        df.to_parquet(directory+'/hospital_data.parquet')
    
        #Remove cache
        try:
            os.remove(directory+'/patients_records_tmp.parquet')
        except:
            pass
    
        #Generate train-validation-test split set 
        generate_train_val_test_sets(df, directory)
    
if __name__ == '__main__':
    
    destination_directory = sys.argv[1]
    data_prop = float(sys.argv[2])
    #Change the following line to the directory containing the parquet's folder and the patient clinic histories'folder
    origin_folder = '/Users/josefinadallavia/Documents/MIM/Tesis/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/data'
    main(destination_directory,origin_folder, data_prop)
    