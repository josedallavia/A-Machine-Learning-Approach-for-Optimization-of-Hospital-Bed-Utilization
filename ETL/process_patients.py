import pandas as pd
import os
import shutil
import sys 

pd.set_option('display.max_columns', None)  
from thesis_lib.data_processing import *
from thesis_lib.visual import *
import warnings
warnings.filterwarnings('ignore')
from datetime import timedelta
from datetime import datetime
from collections import namedtuple
from thesis_lib.hospital_classes import *
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

def main(directory):
    
    #Extract
    print('Fetching data...')
    all_data = load_parquet('parquet_data')
    patients = sorted(
                    [patient for patient 
                    in all_data['hospitalizations'].patient_id.unique()
                    if '/' not in patient
                    ])
                    
    print(str(len(patients))+' patients clinic histories will be processed')
    
    try:
        os.mkdir(directory)
        print(directory, 'was created')
    except:
        pass

    print('Starting patients data processing...')
    
    #Transform
    with tqdm(total=len(patients)) as pbar:
        i = 0
        for patient_id in patients:
            print('Patient_id #:',patient_id)
            filename = directory+'/patient_id_'+patient_id+'.parquet'
            
            if not os.path.isfile(filename):
                patient = Patient(patient_id)
                print('\t Processing patient data')
                patient.load_patient_data(all_data)
                patient_clinic_history = patient.get_historic_records()

                parquet = pa.Table.from_pandas(patient_clinic_history)
            
                #Load
                pq.write_table(parquet, filename )
                print('\t Patient clinic history saved') 
                
            else:
                print('\t clinic history already saved') 
                
            i += 1
            if i % 10 == 0:
                pbar.update(10)
                print(i, 'clinic history files generated')
                print(len(patients)-i, 'patients remaining to be proccessed')

if __name__ == '__main__':
    main(sys.argv[1])
    