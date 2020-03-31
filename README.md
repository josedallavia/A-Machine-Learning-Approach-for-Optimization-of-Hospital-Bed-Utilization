# A-Machine-Learning-Approach-for-Optimization-of-Hospital-Bed-Availability

### Introduction

This project objetive is to build a ML model that allows a hospital to understand its bed availability at a certain point in time by predicting when currently hospitalized patient's will be discharged based on their clinic history as well as environmental factors and historic trends. This  information is crucial for a hospital's management team in order to improve their bed capacity management efficiencies.

PSA: Data will not be published as it was provided by a health center under a Non-Disclosure-Agreement. 

### Table of contents


* [ETL](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/tree/master/ETL)
This folder contains different jobs that take part into the Extract, Transform and Load process of this project. 

  * [preprocess_db.py](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/blob/master/ETL/preprocess_db.py)
  * [clinic_histories.py](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/blob/master/ETL/clinic_histories.py)
  * [dataset.py](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/blob/master/ETL/dataset.py)
  
  The functions used in each of the jobs above described are organized into the [etl_helpers](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/tree/master/ETL/etl_helpers) folder.

Contains the code used for some preprocessing we perfomed on the raw data as well as a the program that allows to generate all  patient historics records. All patients historic records consists of a dataset contanining one row per day a patient was hospitalized with all the available data for that patient upto that very same date (images, surgeries, laboratory analysis and hospital sectors). 

* [thesis_lib](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/tree/master/thesis_lib)

This library contains classes that were exclusively created to work with this project. So far it consists of two modules: 
  * [visual](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/blob/master/thesis_lib/visual.py)
  * [hospital_classes](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/blob/master/thesis_lib/hospital_classes.py)

* [model](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/tree/master/model)

  * [Exploration](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/blob/master/model/Exploratory.ipynb)
  * [MVP](https://github.com/josedallavia/A-Machine-Learning-Approach-for-Prediction-of-Hospital-Bed-Availability/blob/master/model/MVP.ipynb)


In this notebook the different datasets provided by the medical institution are explored with the intention of getting a general understanding of the information we have at our disposal.

