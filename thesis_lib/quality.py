import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt 
import thesis_lib
from datetime import datetime, date 
from thesis_lib.data_processing import *
from thesis_lib.visual import *
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from math import ceil
from numpy import nan


class QualityReport():
    def __init__(self, dataset):
        self.df = dataset.df
        self.categorical = dataset.categorical_variables
        self.numerical = dataset.numerical_variables
        self.df_name = dataset.df_name
        self.categorical_to_plot = self.get_categorical_to_plot()
    

    def get_categorical_to_plot(self):
        
        print('Filtering out categorical columns with max relative frequency < 0.05 and > 0.95')
        categorical_to_plot = []
        for column in self.categorical:
            max_agregation_prop = self.df[column].value_counts(normalize=True)[0]
            if max_agregation_prop > 0.05 and max_agregation_prop < 0.99 :
                    categorical_to_plot.append(column)

        return categorical_to_plot
    
    def get_categorical_plots(self):
        
        print('\t Getting histograms for categorical variables...')
        print('\t\t Categorical variables with max relative frequency < 0.05 or >0.95 will be excluded')
        
        n_cols = 3
        for i in range (len(self.categorical_to_plot)):
            if (i + 3)  % n_cols  == 0:

                fig, axs = plt.subplots(1, 3,figsize=(20,5))
                #plt.tight_layout()
                categorical = self.categorical_to_plot[i:i+3]

                for variable in categorical:
                    for j, ax in enumerate(fig.axes):
                        if j < len(categorical):
                            variable = categorical[j]
                            bars_data = self.df[variable].value_counts(normalize=True)
                            if len(bars_data) > 12:
                                bars_data[:10].plot.bar(ax = ax, title=variable)
                                labels= ax.get_xticklabels()
                                ax.set_xticklabels(labels, rotation=45)

                            else:
                                bars_data.plot.bar(ax = ax, title=variable)
                                labels= ax.get_xticklabels()
                                ax.set_xticklabels(labels, rotation=45)
                        else:
                            ax.axis('off')
        plt.show()
    
    def get_numerical_plots(self):
        
        print('\t Getting distribution plots for numerical variables...')
        
        for numerical in self.numerical:
            fig, axs = plt.subplots(1, 3,figsize=(20,5))
            fig.suptitle(numerical)
            #plt.tight_layout()
            self.df[numerical].plot.hist(ax = axs[0])
            self.df[numerical].plot.density(ax = axs[1])
            self.df[numerical].plot.box(ax = axs[2])
            plt.setp(axs[2].get_xticklabels(), visible=False)
        plt.show()
            
            
   
    def get_tabular_report(self,variable_type):

        """
        variable_type (str) options are: 'numerical' or 'variables'
        """
        print('\t Getting tabular report for {variable_type} variables...'.format(variable_type=variable_type))
        
        
        if variable_type == 'numerical':
            report_columns = ['dataset','variable', 'count', 'mean', 'std', 
                              'min', '25%', '50%', '75%', 'max','missings',
                             'missings %']
            report_df_columns = self.df[self.numerical]
            
        else:
            report_columns = ['dataset','variable','count','unique','top',
                              'freq','missings', 'missings %']
            report_df_columns = self.df[self.categorical]

            
        raw_report = report_df_columns.describe(include='all')
        report = raw_report.T.reset_index().rename(columns={'index':'variable'})
        
        
        len_df = self.df.shape[0]
        report['missings'] = len_df - report['count']
        report['missings %'] = (report['missings']/len_df)*100
        report['dataset'] = self.df_name


        return report[report_columns]
    
    def get_numerical_report(self):
        
        report = self.get_tabular_report('numerical')
        display(HTML(report.to_html()))
        self.get_numerical_plots()
        
    
    def get_categorical_report(self):
        
        report = self.get_tabular_report('categorical')
        display(HTML(report.to_html()))
        self.get_categorical_plots()
        
    def get_report(self, report_type='full'):
        
        print('Getting {report_type} quality report for: {dataset_name} dataset...'.format(report_type=report_type,
                                                                                       dataset_name = self.df_name,
                                                                                       ))
        if report_type == 'full':
            print(' \t CATEGORICAL VARIABLES')
            self.get_categorical_report()
            print(' \t NUMERICAL VARIABLES')
            self.get_numerical_report()
        elif report_type == 'categorical':
            self.get_categorical_report()
        elif report_type == 'numerical':
            self.get_numerical_report()
        else:
            raise ValueError('Invalid report type requested')
            
            
def load_categorical_variables():
    
    categorical_dict = {
        'hospitalizations':  ['admission_id','patient_id','gender','birth_date',
                              'insurance_entity', 'entity_group','admission_date',
                              'admission_time',
                              'admission_year','admission_month', 'origin',
                              'admission_physician',
                              'admission_sector', 'last_sector', 'last_category', 
                              'isolation',
                              'last_room', 'last_bed', 'discharge_date', 'discharge_time',
                              'discharge_year', 'discharge_month','discharge_reason',
                              'pre_discharge_date', 
                              'first_sector', 'administrative_diagnosis','diagnosis_code', 
                              'presumptive_dianogsis','discharge_diagnosis_code',
                              'discharge_diagnosis',
                              'date_registered_discharge','time_registered_discharge', 
                              'discharge_physician',
                              'discharge_summary','discharge_summary_physician','surgery',
                              'express_hip_surgery',
                              'responsible_sector','second_responsible_sector', 
                              'emergency_admission_datetime',
                              'emergency_service', 'has_previous_admission',
                              'previous_admission_id',
                              'previous_admission_date','previous_discharge_date',
                              'previous_sector',
                              'previous_discharge_dianosis','discharge_ambulance',
                              'PIM2TEP', 'high_risk_TEP', 
                              'low_risk_TEP','ARM_TEP','CEC_TEP','request_number',
                              'request_origin','request','request_diagnosis', 
                              'request_sector','notified','request_user','admission_datetime',
                              'discharge_datetime'],
        
        'laboratories':  ['labo_id', 'labo_pun', 'status', 'labo_date', 'labo_time',
                          'labo_year-month', 'sector', 'admission_id', 'patient_id',
                          'admission_date', 'discharge_date', 'entity_id',
                          'insurance_entity','entity_affiliate_id', 'entity_group', 
                          'emergency', 'requester_name','requester_role', 'study_code',
                          'study_description'],
        
        'images': ['image_id','image_pun','status','image_date','image_time',
                   'image_year-month','sector','admission_id','patient_id',
                   'admission_date','discharge_date','entity_id','insurance_entity',
                   'entity_group','emergency','type_of_service','requester_name',
                   'requester_role','study_code','study_description'],
        
        'surgeries': ['operating_room','surgery_date', 'surgery_weekday','surgery_year_month',
                      'surgery_scheduled_time','admission_id','surgery_type','G.A.P.', 
                      'surgery_id','origin','patient_id','gender','discharge_type', 
                      'entity_description','diagnosis','scheduled_surgery','actual_surgery', 
                      'surgery_physician','dependency','anesthesia_type','ASA','antisepsia',
                      'prophylactic_ATB','bed_request','hemotherapy','hemo_ok','x_ray', 
                      'cardiologist','supplies','supplies_ok', 'protocol_no',
                      'service_description','surgery_code','sector_bed','specialization',
                      'surgery_startime', 'surgery_endtime','hospitalization_date',
                      'specialization_code','entry_time','exit_time','anesthesia_startime', 
                      'anesthesia_endtime','post_surgery_condition','admission_date','admission_time',
                      'discharge_date','discharge_time','re_surgery','hips_surgery','injury_condition',
                      'pre_surgery_duration','re_admission_id','antibiotic','anesthetist_id',
                      'nulliparous','new_born_id_rel','new_born_admission_id','new_born_alive',
                      'bact_positive','scheduled_surgery_done'],
        
        'hospital_sectors': ['sector_code','sector_name','sector_type'],
        
        'sectors_admissions': ['admission_id', 'patient_id', 'sector_admission_date',
                               'sector_admission_time', 'sector_code', 'category',
                               'sector_admission_datetime']
    }
    
    return categorical_dict

def load_numerical_variables():
    
    numerical_dict = {
        'hospitalizations':  ['age','admission_lenght_days','new_born_weight',
                              'new_born_gestation_age'],
         
        'laboratories': ['no_of_studies'],
         
        'images': ['no_of_studies'],
         
        'surgeries': ['age','dosis_mg','estimated_duration','surgery_delay',
                      'no_of_surgeries','surgery_duration','post_surgery_duration', 
                      'surgery_prep_duration','no._of_assistans','no._of_pregnancies',
                      'no._births','no._of_cesarean','new_born_weight', 
                      'new_born_gestation_age'],
         
        'hospital_sectors': [],
         
        'sectors_admissions': []}
        
    return numerical_dict