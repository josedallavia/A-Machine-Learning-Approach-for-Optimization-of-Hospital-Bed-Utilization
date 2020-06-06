import os
import pandas as pd
from matplotlib import pyplot as plt   
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_union
import lightgbm as lgb

from lightgbm import LGBMClassifier

from thesis_lib.modelling.data import *
from thesis_lib.modelling.pipeline import *
from thesis_lib.modelling.classifiers import *

class Model():
    def __init__(self, model_params):
        
        self.model_params = model_params
       
        
        self.pipeline = build_pipeline(self.model_params['categorical_features'],
                                       self.model_params['numerical_features'],
                                       self.model_params['accepts_sparse'])
        
    
    @property
    def model_features(self):
        
        feature_names = []
        transformers = self.pipeline.named_steps['feature_engineering'].transformer_list
        for transformer in transformers:
            transformer_features = transformer[1].get_feature_names()
            feature_names.extend(transformer_features)
          
        #HACK
        print(feature_names)
        feature_names = ["".join (c if c.isascii() and c.isalnum() else "_" for c in str(x)) 
                         for x in feature_names]
        
        return feature_names
    
    def transform(self,data):
        
        print('Fitting pipeline...')
        self.pipeline.fit(data.train.X)
        
        print('Transforming data...')
        self.X_train = self.pipeline.transform(data.train.X)
        self.y_train = data.train.y
        
        self.X_val = self.pipeline.transform(data.val.X)
        self.y_val = data.val.y
        
        #self.X_test = self.pipeline.transform(data.test.X)
        #self.y_test = data.test.y
        
    
    def fit_classifier(self, params={}):
    
        
        if self.model_params['classifier'] == 'lgbm':
            self.classifier =  LGBM_classifier(**params)
        elif self.model_params['classifier'] == 'random_forest':
            self.classifier = RFClassifier(**params)
            
        print('Training classifier')   
        self.classifier.fit(self.X_train, self.y_train)
        
        
    def predict(self,X_transf):
        return  self.classifier.predict(X_transf)
    
    def get_feature_importance(self):
        
        return self.classifier.feature_importance_, self.model_features
    
# def plot_feature_importance(self, importance_type='split',n_features=30):
#        
#        lgb.plot_importance(self.classifier,figsize=(10,10),
#                            max_num_features= n_features,
#                           importance_type=importance_type)
#        
    def get_performance_metrics(self):
        
        auc_train = roc_auc_score(self.y_train, self.predict(self.X_train))
        auc_val = roc_auc_score(self.y_val, self.predict(self.X_val))
        
        print('training AUC ROC score: ', auc_train)

        print('validation AUC ROC score: ', auc_val)
        
        print('relative overfitting: ', abs(auc_train-auc_val)/auc_train)
        