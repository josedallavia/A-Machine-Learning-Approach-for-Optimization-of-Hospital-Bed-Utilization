import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt 
from datetime import timedelta
from random import random
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('thesis_lib'))))

def get_clinic_histories(folder, sample_prop=1):   
     
    path= folder+'/'
    df = pd.DataFrame()
    
    print('Will load '+str(sample_prop*100)+'% of data')
    
    for r, d, f in os.walk(path):
        i = 0
        print('Loading patients\' data...')
        for file in f:
            if '.parquet' in file:
                if random() <= sample_prop: 
                    doc = pd.read_parquet(path+file)
                    if i == 0:
                        df = doc
                    else:
                        df = df.append(doc,ignore_index=True)
                    i += 1   
                    if i % 1000 == 0:
                        print('\t Loaded percentage: ',str((i/len(f))*100)+'%')
                           
    return df

def add_static_variables(df, static_level, vars_list, hosp_data):
    
    for var in vars_list:
        var_dict = { data_point[0]: data_point[1] for data_point in  
                         hosp_data[[static_level,var]].groupby([static_level,var]).nunique().axes[0]} 
        df[var] = df[static_level].apply(lambda x: var_dict[x] if x in var_dict else None)
        
    return df

def generate_train_val_test_sets(df, directory):
    
    print('Generating training set')
    train_data = df[df.date < df.date.max()-timedelta(days=365)]
    
    print('Training set:')
    print('\t '+str(len(train_data))+' records') 
    print('\t '+ str( (len(train_data)/len(df))*100 )+' percentage of dataset')
    print('\tFrom: ', train_data.date.min(),'to: ', train_data.date.max())
    
    train_data.to_parquet(directory+'/hospital_train_data.parquet')
    print('Training set saved in: ', directory+'/hospital_train_data.parquet')
    
    test_data = df[df.date >= df.date.max()-timedelta(days=365)]
    test_data['tmp_col'] = pd.Series(np.random.random_sample((len(test_data))))
    
    print('Generating validation set')
    val_data = test_data[test_data['tmp_col'] <= 0.5]
    
    print('Validation set:')
    print('\t'+str(len(val_data))+' records' ) 
    print('\t'+str( (len(val_data)/len(df))*100 )+' percentage of dataset')
    print('\tFrom:', val_data.date.min(), 'to: s',val_data.date.max())
    
    val_data.to_parquet(directory+'/hospital_val_data.parquet')
    print('Validation set saved in: ', directory+'/hospital_val_data.parquet')
    
    print('Generating test set')
    test_data = test_data[test_data['tmp_col'] > 0.5]
    
    print('Test set:')
    print('\t'+ str(len(test_data))+' records' ) 
    print('\t'+ str((len(test_data)/len(df))*100)+' percentage of dataset' )
    print('\tFrom: ',test_data.date.min(),'to: ',test_data.date.max())
    
    test_data.to_parquet(directory+'/hospital_test_data.parquet')
    print('Test set saved in: ', directory+'/hospital_test_data.parquet')