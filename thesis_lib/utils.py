import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt 
from datetime import timedelta
from random import random
import numpy as np
import os, sys
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt 
from etl_helpers.data_processing import *
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('thesis_lib'))))

def process_raw_files(origin_folder, destination_folder):
    
    path = origin_folder+'/'

    try:
        os.mkdir(destination_folder)
    except:
        pass

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
    
    filenames_dict = {'labos': 'laboratory',
                     'images': 'images',
                     'sectores': 'sectors',
                     'internaciones':'hospitalizations',
                     'ingresos_sectores':'sectors_admissions',
                     'cirugias': 'surgeries'}
    
    db = {}
    
    for r, d, f in os.walk(path):
        for file in f:
            if 'csv' in file:
                df_name = filenames_dict[file[:-4]]
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
                    if file == filename[:-8]:
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
            
from numpy import nan
def get_quality_report(df,variables,variable_type,df_name=None):
    
    """
    df: (pandas DataFrame) to get quality report for
    variables: (list of str)variables to include in the report
    df_name: (str) optional
    variable_type (str) options are: 'numerical' or 'variables'
    """
    
    if variable_type == 'numerical':
        report_columns = ['variable', 'count', 'mean', 'std', 
                          'min', '25%', '50%', '75%', 'max','missings']
    else:
        report_columns = ['variable','count','unique','top',
                          'freq','missings']
    
    if variable_type =='categorical':
        for column in variables:
            df[column] = df[column].astype('str')
            df[column].loc[df[column].isin(['NaT','nan','None']) ] = nan
        
    report = df[variables].describe(include='all').T.reset_index().rename(columns={'index':
                                                                                   'variable'})
    
    len_df = df.shape[0]
    report['missings'] = len_df - report['count']
    
    if df_name:
        output_columns = ['dataset']
        output_columns.extend(report_columns)
        report['dataset'] = df_name
    else:
        output_columns = report_columns

    
    return report[output_columns]
    
def load_columns_dict(dataset):
    
    if dataset == 'admissions':
        series_names = {
            'Nro Adm': 'admission_id', 
            'Nro H.C.': 'patient_id', 
            'Edad': 'age', 
            'Sexo': 'gender', 
            'Fec Nac': 'birth_date', 
            'Entidad': 'insurance_entity',
            'AgrupEntidad': 'entity_group' , 
            'Fec Adm': 'admission_date', 
            'Hora Adm': 'admission_time', 
            'AñoAdm': 'admission_year', 
            'MesAdm': 'admission_month',
            'Procedencia': 'origin', 
            'Médico Admisión': 'admission_physician', 
            'SecAdmisión': 'admission_sector', 
            'SecUltimo': 'last_sector',
            'CategUlt': 'last_category', 
            'Aislación': 'isolation', 
            'HabitUlt': 'last_room', 
            'CamaUlt': 'last_bed', 
            'Fec Alta': 'discharge_date', 
            'Hora Alta': 'discharge_time',
            'AñoAlta': 'discharge_year', 
            'MesAlta': 'discharge_month', 
            'Motivo Alta': 'discharge_reason', 
            'PreAlta': 'pre_discharge_date', 
            '1erSecInt': 'first_sector',
            'Diagnóstico Administrativo No Codificado': 'administrative_diagnosis', 
            'CodDiagPresu': 'diagnosis_code',
            'Diagnóstico Presuntivo': 'presumptive_dianogsis',
            'CodCieDiagEgr': 'discharge_diagnosis_code', 
            'Diagnóstico Egreso': 'discharge_diagnosis',
            'FecDenunEgre': 'date_registered_discharge', 
            'HoraDenunEgre': 'time_registered_discharge', 
            'Médico Denuncia Egreso': 'discharge_physician', 
            'Epicrisis': 'discharge_summary',
            'MédicoEpicrisis': 'discharge_summary_physician', 
            'Quirurg': 'surgery', 
            'CaderaExpress': 'express_hip_surgery',
            'Permanencia': 'admission_lenght_days',
            'Servicio Responsable': 'responsible_sector',
            'Servicio Co-Responsable': 'second_responsible_sector', 
            'FecHorIngGua': 'emergency_admission_datetime',
            'Prest.Guardia': 'emergency_service', 
            'Reingreso': 'has_previous_admission', 
            'AdmAntReciente':  'previous_admission_id' ,
            'FecAdmAntReciente': 'previous_admission_date', 
            'FecAltaAntReciente': 'previous_discharge_date', 
            'SecAntReciente': 'previous_sector',
            'Diagn. Egreso Admisión Anterior Reciente': 'previous_discharge_dianosis', 
            'AmbulanciaEgreso': 'discharge_ambulance',
            'PesoAlNacer': 'new_born_weight', 
            'EdadGestac': 'new_born_gestation_age',
            'PIM2TEP': 'PIM2TEP', 
            'DiagAltoRiesgoTEP': 'high_risk_TEP',
            'DiagBajoRiesgoTEP': 'low_risk_TEP', 
            'ARM_TEP': 'ARM_TEP', 
            'CEC_TEP': 'CEC_TEP', 
            'SolicDerivación': 'request_number',
            'OrigDerivación': 'request_origin',
            'Procedencia.1': 'request', 
            'DiagnósticoDerivación': 'request_diagnosis',
            'AreaDerivación': 'request_sector',
            'Notificado': 'notified', 
            'UsuarioDeriv': 'request_user'}
    elif dataset == 'laboratory':
        series_names = { 
        'Nro Vale': 'labo_id',
        'Pun': 'labo_pun',
        'Estado': 'status',
        'Fecha': 'labo_date', 
        'Hora': 'labo_time',
        'AñoMes':'labo_year-month',
        'Sector': 'sector',
        'Nro Adm': 'admission_id',
        'HistClín': 'patient_id',
        'Fec.Adm.': 'admission_date',
        'Fec.Alta': 'discharge_date',
        'Entidad': 'entity_id',
        'Nombre Entidad': 'insurance_entity',
        'Nro.Afiliado': 'entity_affiliate_id',
        'GrupoEnt': 'entity_group',
        'Urgencia': 'emergency',
        'Nombre del Solicitante': 'requester_name',
        'Función del Solicitante': 'requester_role',
        'Prestación': 'study_code',
        'Descrip Prestación': 'study_description',
        'CantPrest': 'no_of_studies',
        }
    elif dataset == 'images':
        series_names = { 
        'Nro Vale': 'image_id',
        'Pun': 'image_pun',
        'Estado': 'status',
        'Fecha': 'image_date', 
        'Hora': 'image_time',
        'AñoMes':'image_year-month',
        'Sector': 'sector',
        'Nro Adm': 'admission_id',
        'HistClín': 'patient_id',
        'Fec.Adm.': 'admission_date',
        'Fec.Alta': 'discharge_date',
        'Entidad': 'entity_id',
        'Nombre Entidad': 'insurance_entity',
        'Nro.Afiliado': 'entity_affiliate_id',
        'GrupoEnt': 'entity_group',
        'Urgencia': 'emergency',
        'Nombre del Solicitante': 'requester_name',
        'Función del Solicitante': 'requester_role',
        'Prestación': 'study_code',
        'Descrip Prestación': 'study_description',
        'CantPrest': 'no_of_studies',
         'Servicio': 'type_of_service'}
        
    elif dataset == 'surgeries':
        
        series_names = {
            'Quirófano' : 'operating_room', 
            'FechaQuirof' : 'surgery_date', 
            'DíaSem': 'surgery_weekday', 
            'AñoMes': 'surgery_year_month',
            'HoraProgram': 'surgery_scheduled_time', 
            'CodAdmision': 'admission_id', 
            'UrgProgProtoc': 'surgery_type',
            'IdCirugia': 'surgery_id',
            'DescProgOri': 'origin' ,
            'HistClínica': 'patient_id',
            'Sexo': 'gender', 
            'Edad': 'age', 
            'Motivo Alta': 'discharge_type',
            'DescripEntidad': 'entity_description', 
            'Diagnostico': 'diagnosis',
            'Operación Programada': 'scheduled_surgery',
            'Operación en Parte Quirúrgico': 'actual_surgery', 
            'Cirujano': 'surgery_physician', 
            'Dependencia': 'dependency',
            'DescripTipoAnestesia': 'anesthesia_type',  
            'Antisepsia': 'antisepsia', 
            'ATB profiláctico':'prophylactic_ATB',
            'Dosis mg': 'dosis_mg', 
            'CamaSolic': 'bed_request', 
            'Hemoterapia': 'hemotherapy' , 
            'HemoOk': 'hemo_ok', 
            'Rayos': 'x_ray' , 
            'Cardiologo':'cardiologist' ,
            'Material': 'supplies',
            'MateOk': 'supplies_ok', 
            'NroProtocolo': 'protocol_no', 
            'DescripServicio': 'service_description', 
            'OperCod': 'surgery_code',
            'CamaSector': 'sector_bed', 
            'DescripEspecialidad':'specialization',
            'DuracionEstimada':'estimated_duration', 
            'HoraInic': 'surgery_startime',
            'HoraFin': 'surgery_endtime', 
            'DemoraInicio': 'surgery_delay',
            'FechaInternac': 'hospitalization_date', 
            'EspecPrestador': 'specialization_code',
            'HoraIngre': 'entry_time' , 
            'HoraEgre': 'exit_time', 
            'HoraIndAnes': 'anesthesia_startime', 
            'HoraFinAnes': 'anesthesia_endtime', 
            'PostOper':  'post_surgery_condition',
            'FechaAdmision' : 'admission_date', 
            'HoraAdm': 'admission_time', 
            'FechaAlta': 'discharge_date',
            'HoraAlta' : 'discharge_time', 
            'ReCirug': 're_surgery', 
            'CantCirug': 'no_of_surgeries', 
            'CaderaExpress': 'hips_surgery',
            'TipoHerida': 'injury_condition' ,
            'DuracionPreCirugia': 'pre_surgery_duration',
            'DuracionCirugia': 'surgery_duration',
            'DuracionPostCirugia': 'post_surgery_duration',
            'DuracionPreparacion': 'surgery_prep_duration', 
            'Reinternación': 're_admission_id',
            'Antibiot':'antibiotic' ,
            'Cultivo': 'seeding', 
            'CntAyud': 'no._of_assistans', 
            'MatricAnest': 'anesthetist_id', 
            'Gestac': 'no._of_pregnancies' , 
            'Partos': 'no._births',
            'Cesár.': 'no._of_cesarean', 
            'Nulípara':'nulliparous', 
            'IdRelRN': 'new_born_id_rel', 
            'NroAdmRN': 'new_born_admission_id', 
            'PesoRecNac': 'new_born_weight', 
            'EdadGestac': 'new_born_gestation_age', 
            'BebeEgresoVivo': 'new_born_alive', 
            'BactPosit': 'bact_positive'}
    elif dataset == 'hospital_sectors':
        series_names = { 
            'CodSector': 'sector_code',
            'NombreSector': 'sector_name',
            'TipoSector': 'sector_type'}
        
    elif dataset == 'sectors_admissions':
        series_names = { 
           'Nro Adm': 'admission_id', 
            'Nro H.C.': 'patient_id', 
            'FecIngrSec': 'sector_admission_date', 
            'HoraIngrSec': 'sector_admission_time', 
            'Sector': 'sector_code', 
            'Categ': 'category'}
    else:
        series_names = {}
    
    return series_names
        
def load_boolean_cols(dataset):
    if dataset == 'admissions':
        boolean_cols = ['discharge_summary',
                        'surgery',
                        'express_hip_surgery',
                        'has_previous_admission',
                        'discharge_ambulance',
                        'ARM_TEP',
                        'CEC_TEP']
    elif dataset == 'surgeries':
        boolean_cols = ['G.A.P.',
                        'bed_request', 
                        'hemotherapy', 
                        'hemo_ok', 
                        'x_ray', 
                        'cardiologist', 
                        'supplies',
                        'supplies_ok',
                        'hips_surgery',
                        'nulliparous']
    else:
        boolean_cols = []
        
    return boolean_cols

def load_cols_to_drop(dataset):
    
    if dataset == 'admissions':
        columns_to_drop = ['Prestac.Guardia']
    elif dataset == 'laboratory':
        columns_to_drop = ['Nombre Paciente',
                           'HH',
                           'Mnemo Serv',
                           'Cod Serv',
                           'Cod Insumo',
                           'Descrip Insumo',
                           'CantInsumos',
                           'EstadoResultado',
                           'Observ.Estudio']
    elif dataset == 'images':
        columns_to_drop = ['HH',
                           'Mnemo Serv',
                           'Cod Serv',
                           'Cod Insumo',
                           'Descrip Insumo',
                           'CantInsumos',
                           'Observ.Estudio']
    elif dataset == 'surgeries':
        columns_to_drop = ['EstadoAgrup',
                           'DíasIntern',
                           'HhIndAnes',
                           'HhInicio',
                           'TorreLapar']
    else:
        columns_to_drop = []
    
    return columns_to_drop

def rename_columns(df, dataset_name):
    series_names = load_columns_dict(dataset_name)
    df.rename(series_names,axis=1, inplace=True)
    
    return df

def cast_boolean_cols(df,dataset_name):
    boolean_cols = load_boolean_cols(dataset_name)
    
    if len(boolean_cols) > 0:
        for col in boolean_cols:
            df[col] = df[col].apply(lambda x : True if x == 'Sí' else  (False if x == 'No' else None ))
    return df

def drop_cols(df, dataset_name):
    
    columns_to_drop = load_cols_to_drop(dataset_name)
    
    if len(columns_to_drop) > 0:
        df.drop(columns_to_drop,axis=1, inplace=True)
        
    return df

def generate_new_cols(df,dataset_name):
    
    if dataset_name == 'admissions':
        
        df['admission_datetime'] = pd.to_datetime(df['admission_date'].map(str)+" "+
                                                  df['admission_time'].map(str)
                                                 )
        df['discharge_datetime'] = pd.to_datetime(
                                            df['discharge_date'].map(str)+" "+
                                            df['discharge_time'].map(str),
                                            errors='ignore'
                                            )
    elif dataset_name == 'sectors_admission':
        
        df['sector_admission_datetime']  = pd.to_datetime(
                                                df.sector_admission_date.map(str)+" "+
                                                df.sector_admission_time.map(str)
                                                )
    elif dataset_name == 'surgeries':
        
        duration_cols = ['estimated_duration',
                         'pre_surgery_duration',
                         'surgery_duration',
                         'post_surgery_duration',
                         'surgery_prep_duration']
        
        for col in duration_cols:
            minutes_duration_series = []
            for i in pd.to_datetime(df[col], errors='ignore').dt.time:
                minutes_duration_series.append(i.hour*60+i.minute)

            df[col] = pd.Series(minutes_duration_series)
        
        df['scheduled_surgery_done'] = df.scheduled_surgery == df.actual_surgery
        
    
    return df

def process_df(df,dataset_name):
    
    print('Procesing dataset: ',dataset_name)
    print('\t Renaming columns')
    df = rename_columns(df, dataset_name)
    print('\t Castin boolean columns')
    df = cast_boolean_cols(df, dataset_name)
    print('\t Dropping unnecesary columns')
    df = drop_cols(df, dataset_name)
    print('\t Generating some new cols')
    df = generate_new_cols(df,dataset_name)
    
    if dataset_name == 'admissions':
        df = df.loc[~df['discharge_date'].isna()]
    
    return df
    
    


    