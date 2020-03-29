import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt 
from datetime import timedelta
from random import random
import numpy as np




def process_raw_files(origin_folder, destination_folder):
    
    path = origin_folder+'/'

    try:
        os.mkdir(destination_folder)
    except:
        pass

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.DS_Store' not in file:
                print(file)
                if 'xlsx' in file:
                    
                    #Read excel file
                    df = pd.read_excel(os.path.join(r, file))
                    
                    #Convert to csv with ',' separator
                    new_file_name = destination_folder+'/'+file[:-4]+'csv'
                    df.to_csv(new_file_name, sep=',', index=False)
                    
                    print(file+' successfully converted to csv: '+new_file_name)

                elif 'csv' in file:
                    
                    #Read csv file 
                    df = pd.read_csv(os.path.join(r, file), sep=';')
                    
                    #Convert to csv with ',' separator
                    new_file_name = destination_folder+'/'+file
                    df.to_csv(new_file_name, sep=',', index=False)
                    print(file+' successfully copied to '+new_file_name)

def get_database(folder):    
    path= folder+'/'
    db = {}
    
    for r, d, f in os.walk(path):
        for file in f:
            if 'csv' in file:
                df_name = file[:-4]
                print('Loading dataset: ',df_name)
                db[df_name] = pd.read_csv(path+file)
    
    
    return db
    
    
def get_variables_ref_lists(db):
    
    directory = "VariableReferences"
  
  
    # Path 
    path = os.path.join(os.getcwd(), directory) 

    try:
        os.mkdir(path)
    except:
        pass

    for table_name in db.keys():
        variables = open(str(path)+'/variables_'+table_name+'.txt', 'w')
        variables.write('variable_name, variable_definition'+'\n')
        for column in db[table_name]:
            variables.write(str(column)+',"" \n')

        variables.close()
        
def load_columns(column_type):
    
    if column_type == 'integer':
        return  ['no_of_studies','age','admission_year','discharge_year',
                'admission_lenght_days','new_born_gestation_age','surgery_delay',
                'no_of_surgeries','no._of_assistans', 'no._of_pregnancies',
                'no._births', 'no._of_cesarean', 'estimated_duration',
                 'pre_surgery_duration','surgery_duration','post_surgery_duration',
                 'surgery_prep_duration']
    
    elif column_type == 'date':
        return ['labo_date', 'admission_date', 'discharge_date','image_date',
                 'birth_date','pre_discharge_date','date_registered_discharge',
                 'emergency_admission_datetime','previous_admission_date',
                 'previous_discharge_date','surgery_date',
                 'sector_admission_date']
    
    elif column_type == 'time':   
        return ['labo_time','image_time','admission_time','discharge_time',
                 'time_registered_discharge','surgery_scheduled_time', 
                 'surgery_startime','surgery_endtime',
                 'entry_time', 'exit_time','anesthesia_startime', 
                 'anesthesia_endtime',
                 'sector_admission_time']
    
    elif column_type == 'datetime':
         return ['sector_admission_datetime','admission_datetime','discharge_datetime']

def format_integer_col(df,col):
    
    return df[col].fillna(0.0).astype(int,errors='ignore')

def format_date_col(df,col,date_format=None):
    
    if date_format != None: 
        return pd.to_datetime(df[col] , errors='ignore', format=date_format).dt.date
    else:
        return pd.to_datetime(df[col] , errors='ignore').dt.date
    
def format_datetime_col(df,col):
    
    return pd.to_datetime(df[col] , errors='ignore')
  
  
        
def format_time_col(df,col):
    
    return pd.to_datetime(df[col], errors='ignore').dt.time

def load_parquet(parquets_folder,file=None):    
    
    path= parquets_folder+'/'
    
    db = {}
    
    for r, d, f in os.walk(path):
        f = [file for file in f if '.parquet' in file]
        if len(f) == 0: 
            print('No files in the parquets folder to load!!! ')
        else:
            for filename in f:
                if not file:
                    df_name = filename[:-8]
                    print('Loading dataset: ',df_name)
                    db[df_name] = pd.read_parquet(path+filename)
                    print('\t',len(db[df_name]))
                        
                else:
                    if file in filename:
                        df_name = file
                        print('Loading dataset: ',df_name)
                        db[df_name] = pd.read_parquet(path+filename)
                        print('\t',len(db[df_name]))
                            
    
            print('Formating integer columns')
            for col in load_columns('integer'):
                for df in db:
                    if col in db[df].columns:
                        db[df][col] = format_integer_col(db[df],col)
          

            print('Formating date columns')
            for col in load_columns('date'):
                for df in db:
                    if col in db[df].columns:
                        if df == 'laboratory':
                            db[df][col] = format_date_col(db[df],col,date_format="%d/%m/%Y")
                        else: 
                            db[df][col] = format_date_col(db[df],col)
    
            print('Formating time columns')
            for col in load_columns('time'):
                for df in db:
                    if col in db[df].columns:
                        db[df][col] = format_time_col(db[df],col)
                
            print('Formating datetime columns')
            for col in load_columns('datetime'):
                for df in db:
                    if col in db[df].columns:
                        db[df][col] = format_datetime_col(db[df],col)
    
    
            return (db if not file else db[file])
    
        
def cast_list(list_to_cast):
    str_list = [str(i) for i in list_to_cast]
    
    return (',').join(str_list)
            
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
    #print('Validation set saved in: ', directory+'/hospital_val_data.parquet')
    
    
    print('Generating test set')
    test_data = test_data[test_data['tmp_col'] > 0.5]
    
    print('Test set:')
    print('\t'+ str(len(test_data))+' records' ) 
    print('\t'+ str((len(test_data)/len(df))*100)+' percentage of dataset' )
    print('\tFrom: ',test_data.date.min(),'to: ',test_data.date.max())
    
    test_data.to_parquet(directory+'/hospital_test_data.parquet')
    print('Test set saved in: ', directory+'/hospital_test_data.parquet')

    