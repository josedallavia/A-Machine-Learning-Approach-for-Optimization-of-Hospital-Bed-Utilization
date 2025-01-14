# A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability

### Introduction

This project objetive is to build a ML model that allows a hospital to understand its bed availability at a certain point in time by predicting when currently hospitalized patient's will be discharged based on their clinic history as well as environmental factors and historic trends. This  information is crucial for a hospital's management team in order to improve their bed capacity management efficiencies.

PSA: Data will not be published as it was provided by a health center under a Non-Disclosure-Agreement. 

### Table of contents


* [ETL](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/tree/master/ETL):
This folder contains the different jobs that take part into the Extract, Transform and Load process of this project. 

  * [preprocess_db](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/blob/master/ETL/preprocess_db.py): This job takes the raw database, as provided by the medical institution, and process it in several ways in order to have a more scalable, consistent and clean data for our project. 
   
  * [clinic_histories](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/blob/master/ETL/clinic_histories.py): This job reads the .parquet files generated by the previous job and for each patient in the database, generates his daily records with his information point-in-time-correct (studies done, diganosis received, time spent at the hospital, physicians who have treated him, etc.). The records are saved for each patient under a patient id named file.
  
  * [dataset](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/blob/master/ETL/dataset.py): This jobs reads patients historic records and build a dataset by combining all patient historic records with other time independent variables as patient gender, and patient idependent variables as day of the week. It then splits the created dataset into three sets: training, validation and testing. This partition is done in a time basis, aiming to replicate the conditions of the model application.
  
  The functions used in each of the jobs above described are organized into the [etl_helpers](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/tree/master/ETL/etl_helpers) folder.


* [thesis_lib](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/tree/master/thesis_lib): 
This library contains modules that were exclusively created to work with different aspects of this project.
     
* [Experiments](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/tree/master/experiments) This folder contains the different modeling approaches experimented in this project rendered in Jupyter Notebooks.





