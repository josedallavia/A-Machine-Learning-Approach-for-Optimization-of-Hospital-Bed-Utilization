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
    all_data = load_parquet('parquet_data')
    patients = sorted(all_data['hospitalizations'].patient_id.unique())
    

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
            patient = Patient(patient_id)
            patient.load_patient_data(all_data)
            patient_clinic_history = patient.get_historic_records()

            parquet = pa.Table.from_pandas(patient_clinic_history)
            
            #Load
            pq.write_table(parquet, directory+'/patient_id_'+patient_id+'.parquet')
            print(patient_id+' clinic history saved') 
            
            i += 1
            if i % 100 == 0:
                pbar.update(100)

if __name__ == '__main__':
    main(sys.argv[1])
    