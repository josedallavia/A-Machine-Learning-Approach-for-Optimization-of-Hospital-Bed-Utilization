import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt 



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
                'no._births', 'no._of_cesarean']
    elif column_type == 'date':
        return ['labo_date', 'admission_date', 'discharge_date','image_date',
                 'birth_date','pre_discharge_date','date_registered_discharge',
                 'emergency_admission_datetime','previous_admission_date',
                 'previous_discharge_date','surgery_date',
                 'sector_admission_date']
    elif column_type == 'time':
        
        return ['labo_time','image_time','admission_time','discharge_time',
                 'time_registered_discharge','surgery_scheduled_time', 
                 'surgery_startime','surgery_endtime','estimated_duration',
                 'entry_time', 'exit_time','anesthesia_startime', 
                 'anesthesia_endtime','pre_surgery_duration','surgery_duration',
                 'post_surgery_duration', 'surgery_prep_duration',
                 'sector_admission_time']

def format_integer_col(df,col):
    
    return df[col].fillna(0.0).astype(int,errors='ignore')

def format_datetime_col(df,col,date_format=None):
    
    if date_format != None: 
        return pd.to_datetime(df[col] , errors='ignore', format=date_format).dt.date
    else:
        return pd.to_datetime(df[col] , errors='ignore').dt.date

        
def format_time_col(df,col):
    
    return pd.to_datetime(df[col], errors='ignore').dt.time

def load_parquet(parquets_folder,file=None):    
    
    path= parquets_folder+'/'
    
    db = {}
    
    for r, d, f in os.walk(path):
            for filename in f:
                if '.parquet' in filename:
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
                    db[df][col] = format_datetime_col(db[df],col,date_format="%d/%m/%Y")
                else: 
                    db[df][col] = format_datetime_col(db[df],col)
    
    print('Formating time columns')
    for col in load_columns('time'):
        
        for df in db:
            if col in db[df].columns:
                
                db[df][col] = format_time_col(db[df],col)
    
    
    return (db if not file else db[file])
    
        
        
        

    