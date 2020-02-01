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
                print(df_name)
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
    