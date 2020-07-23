import pandas as pd

from thesis_lib.utils import *
from thesis_lib.visual import *

from datetime import timedelta
from datetime import datetime
from collections import namedtuple


class Patient:
    def __init__(self, patient_id):
        self.patient_id = patient_id

    def load_patient_data(self, db):
        
        hosp = db['hospitalizations']
        labo = db['laboratory']
        imag  = db['images']
        surg = db['surgeries']
        sect = db['sectors_admissions']
        
        
        self.hospitalizations_data = (
            hosp[hosp['patient_id'] == self.patient_id].sort_values(by='admission_date'))
        self.laboratory_data = (
            labo[labo['patient_id'] == self.patient_id].sort_values(by=['admission_date', 'labo_date']))
        self.images_data = (
            imag[imag['patient_id'] == self.patient_id].sort_values(by=['admission_date', 'image_date']))
        self.surgeries_data = (
            surg[surg['patient_id'] == self.patient_id].sort_values(by=['admission_date', 'surgery_date']))
        self.sectors_data = (
            sect[sect['patient_id'] == self.patient_id].sort_values(by=['admission_id', 'sector_admission_date']))
    
    @property
    def admission_history(self): 
        
        admission_key = namedtuple('admission_key', ['admission_number', 'admission_id'])
        
        if hasattr(self, 'hospitalizations_data'):
            return {admission_key(admission_number,admission_id) : Admission(admission_id, self) 
                   for admission_number, admission_id in enumerate(self.hospitalizations_data.admission_id)}
        else:
            print('Load patient data before')
    
    def get_historic_records(self):

        records_history = pd.DataFrame()
        
        patient_records = {'patient_id': self.patient_id}
        
        for admission in self.admission_history.keys():
            patient_records['admission_id'] = admission.admission_id
            patient_records['no._of_admission'] = admission.admission_number
            
            admission_records = self.admission_history[admission].get_admission_records(patient_records)
            
            if records_history.empty:
                records_history = admission_records
            else:
                records_history = records_history.append(admission_records, ignore_index=True)
        
        return records_history
            
class Admission():
    def __init__(self, admission_id, patient):
        self.admission_id = admission_id 
        self.load_admission_data(patient)

    def load_admission_data(self,patient): 
        
        self.admission_data = self.process_hospitalization_data(patient)
        self.discharge_datetime = datetime.strptime(
            str(self.admission_data['discharge_datetime']), '%Y-%m-%d %H:%M:%S')
        self.admission_datetime = datetime.strptime(
            str(self.admission_data['admission_datetime']), '%Y-%m-%d %H:%M:%S')
        self.laboratory_data = self.process_laboratory_data(patient)                                         
        self.images_data = self.process_images_data(patient)
        self.surgeries_data = self.process_surgeries_data(patient)
        self.sectors_data = self.process_sectors_data(patient)

    def process_hospitalization_data(self, patient):
        return patient.hospitalizations_data[
            (patient.hospitalizations_data.admission_id == self.admission_id)].to_dict(orient='records')[0]
        
    def process_laboratory_data(self, patient):
        return patient.laboratory_data[patient.laboratory_data.admission_id == self.admission_id]

    def process_images_data(self, patient):
        return patient.images_data[patient.images_data.admission_id == self.admission_id]
    
    def process_surgeries_data(self, patient):
        return patient.surgeries_data[patient.surgeries_data.admission_id == self.admission_id] 
    
    def process_sectors_data(self, patient):
        adm_sectors_data = patient.sectors_data[patient.sectors_data.admission_id == self.admission_id]
        
        if len(adm_sectors_data) == 0:
            adm_sectors_data = pd.DataFrame([{
                'admission_id': self.admission_id, 
                'patient_id': patient.patient_id, 
                'sector_admission_date': self.admission_data['admission_date'],
                'sector_admission_time': self.admission_data['admission_time'],
                'sector_code': None, 
                'category': None,
                'sector_admission_datetime': pd.to_datetime(
                    self.admission_data['admission_datetime'], errors='ignore')
            }])
                                
        adm_sectors_data['sector_stay'] = (
            adm_sectors_data['sector_admission_datetime'].diff(periods=1).astype('timedelta64[m]').shift(-1))
        last_sector_stay = (self.discharge_datetime - adm_sectors_data.iloc[-1].sector_admission_datetime)
        
        adm_sectors_data.loc[adm_sectors_data.index[-1], 'sector_stay'] = int(last_sector_stay.seconds/3600)
        
        return adm_sectors_data
    
    def get_labos_date_records(self,date):
        
        labos_records = {}
        
        mask = self.laboratory_data['labo_date'] == date
        labos = self.laboratory_data[mask]

        labos_records['labos_studies_names'] =  cast_list(labos.study_description.unique())
        labos_records['labos_set_count'] = labos.labo_id.nunique()
        labos_records['labos_count'] = len(labos.labo_id)
        labos_records['labos_requester_roles'] = cast_list(labos.requester_role.unique())
        labos_records['labos_requester_roles_count'] = labos.requester_role.nunique()
        labos_records['labos_emergencies'] = labos.emergency.sum()
        labos_records['labos_requesters'] = cast_list(labos.requester_name.unique())
        labos_records['labos_requesters_count'] = labos.requester_name.nunique()
        
        return labos_records 
    
    def get_images_date_records(self,date):
        
        images_records = {}

        mask = self.images_data['image_date'] == date
        images = self.images_data[mask]

        images_records['images_count'] =  images.image_id.nunique()
        images_records['images_study_types'] =  cast_list(images.type_of_service.unique())
        images_records['images_study_types_count'] =  images.type_of_service.nunique()
        images_records['images_studies_names'] = cast_list(images.study_description.unique())
        images_records['images_requesters'] = cast_list(images.requester_name.unique())
        images_records['images_requesters_count'] = images.requester_name.nunique()
        images_records['images_requester_roles'] = cast_list(images.requester_role.unique())
        images_records['images_requester_roles_count'] = images.requester_role.nunique()
        images_records['images_emergencies'] = images.emergency.sum()
    
        return images_records 
    
    def get_surgeries_date_records(self,date):
        
        surgeries_records = {}
        
        mask = self.surgeries_data['surgery_date'] == date
        surgeries = self.surgeries_data[mask]

        surgeries_records['surgeries_count'] =  len(surgeries.surgery_id.values)
        surgeries_records['surgeries_types'] =  cast_list(surgeries.surgery_type.values)
        surgeries_records['surgeries_types_count'] =  surgeries.surgery_type.nunique()
        surgeries_records['surgeries_scheduled'] =  cast_list(surgeries.scheduled_surgery.values)
        surgeries_records['surgeries_scheduled_done'] = cast_list(surgeries.scheduled_surgery_done.values)
        surgeries_records['surgeries_actual'] = cast_list(surgeries.actual_surgery.values)
        surgeries_records['surgeries_pre_surgery_duration'] = surgeries.pre_surgery_duration.sum()
        surgeries_records['surgeries_surgery_duration'] = surgeries.surgery_duration.sum()
        surgeries_records['surgeries_post_surgery_duration'] = surgeries.post_surgery_duration.sum()
        surgeries_records['surgeries_prep_duration'] = surgeries.surgery_prep_duration.sum()
        surgeries_records['surgeries_surgery_delay'] = surgeries.surgery_delay.sum()
        surgeries_records['surgeries_injury_condition'] = cast_list(surgeries.injury_condition.values)
        surgeries_records['surgeries_post_surgery_condition'] = cast_list(surgeries.post_surgery_condition.values)
        surgeries_records['surgeries_services'] = cast_list(surgeries.service_description.values)
        surgeries_records['surgeries_services_count'] =  surgeries.service_description.nunique()
        surgeries_records['surgeries_anesthesia_types'] =  cast_list(surgeries.anesthesia_type.values)
        surgeries_records['surgeries_bact_positive'] =  cast_list(surgeries.bact_positive.values)

        return surgeries_records 
    
    def get_date_sectors(self, date):
        
        if len(self.sectors_data) > 0:
            mask = self.sectors_data['sector_admission_date'] == date
            sectors = self.sectors_data[mask]

            if len(sectors) > 0:
                rv = sectors
            else: 
                rv = self.get_date_sectors(date-timedelta(days=1))
                
            return rv
            
        else:
            print('No sectors data available!')
            
    
    def get_sectors_date_records(self,date):
        
        sectors_records = {}
        sectors = self.get_date_sectors(date)
        
        sectors_records['sectors_last_sector'] = sectors.sector_code.values[-1]
        sectors_records['sectors_last_stay'] = sectors.sector_stay.values[-1]
        
        if sectors.iloc[-1].sector_admission_date == date:
            sectors_records['sectors_count'] = sectors.sector_code.nunique()
            sectors_records['sectors_names'] = cast_list(sectors.sector_code.values)
            sectors_records['sector_stay'] = cast_list(sectors.sector_stay.values)
        else:
        
            sectors_records['sectors_count'] = 0
            sectors_records['sectors_names'] = ''
            sectors_records['sector_stay'] = ''
    
        return sectors_records 
        
    def get_admission_records(self, row):
        
        length_of_stay = (
            self.admission_data['discharge_date'] - self.admission_data['admission_date']).days
        df = pd.DataFrame()
        
        row['labos_cumulative'] =  0
        row['images_cumulative'] =  0
        row['labos_set_cumulative'] = 0
        row['surgeries_cumulative'] = 0
          
        for day in range(length_of_stay):
            row['date'] = self.admission_data['admission_date'] + timedelta(day)
            row['hosp_day_number'] =  day
            row['discharge'] =  (True if day == (length_of_stay -1) else False)
            #Labos 
            labos_records = self.get_labos_date_records(row['date'])
            row.update(labos_records)
            row['labos_cumulative'] += row['labos_count']
            row['labos_set_cumulative'] += row['labos_set_count']
            #Images
            images_records = self.get_images_date_records(row['date'])
            row.update(images_records)
            row['images_cumulative'] += row['images_count']
            #Surgeries
            surgeries_records = self.get_surgeries_date_records(row['date'])
            row.update(surgeries_records)
            row['surgeries_cumulative'] += row['surgeries_count']
            #Sectors
            sectors_records = self.get_sectors_date_records(row['date'])
            row.update(sectors_records)
            
            df = df.append(row, ignore_index=True)
            
        return df