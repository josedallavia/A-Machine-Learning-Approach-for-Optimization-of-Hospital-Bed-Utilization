import os
import pandas as pd
from matplotlib import pyplot as plt   
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_union
import lightgbm as lgb
from thesis_lib.modelling.data import * 


class FeaturePreProcessor():
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
        
        
    def transform(self,X):
        X['admission_weekday'] = pd.to_datetime(X['admission_date']).dt.weekday.astype('str')
        X['date_weekday'] = pd.to_datetime(X['date']).dt.weekday.astype('str')
        return X

class ItemSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, keys,is_categorical=False):
        self.keys = keys
        self.is_categorical = is_categorical

    def fit(self, X, y=None):
        return self

    def transform(self, X):     
        return X[self.keys].astype(str) if self.is_categorical else X[self.keys]
    
class CustomScaler(StandardScaler):
    
    def __init__ (self, with_mean=True,with_std=True):
        super().__init__(with_mean=with_mean, 
                         with_std=with_std)
    
    def fit(self,X,y=None):
        self.feature_names = list(X.columns)
        return super().fit(X)
    
    def transform(self,X):
        return super().transform(X)
        
    def get_feature_names(self):
        return self.feature_names

class CustomEncoder(OneHotEncoder):
    
    def __init__(self,accepts_sparse=True):
        self.accepts_sparse= accepts_sparse
        super().__init__(handle_unknown='ignore',sparse=accepts_sparse)
        
    def fit(self,X,y=None):
        self.features_headers = list(X.columns)
        return super().fit(X)
        
    def get_feature_names(self):
        return super().get_feature_names(self.features_headers)
    
class FeatureProcessor(Pipeline):
    def __init__(self, features_list, transformer_type,accepts_sparse=True):
        self.features_list = features_list
        self.transformer_type = transformer_type
        self.accepts_sparse = accepts_sparse

        self.transformer_dict = {'categorical': CustomEncoder(self.accepts_sparse),
                                 'numerical': CustomScaler()}


        self.selector = ItemSelector(keys= self.features_list,
                                     is_categorical= (self.transformer_type == 'categorical'))
        
        self.transformer = self.transformer_dict[self.transformer_type]
        
        super().__init__([('selector',self.selector),
                          (self.transformer_type, self.transformer)])
        
    def get_feature_names(self):
        return self.transformer.get_feature_names()

class CustomPipeline():
    def __init__(self, categorical_features,
                 numerical_features,
                 accepts_sparse=True):

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.accepts_sparse= accepts_sparse

    def build_pipeline(self):
        #First step of pipeline: preprocesing
        self.preprocessing = ('preprocessing', FeaturePreProcessor())

        #Second step: feature engineering
        numerical_processor = FeatureProcessor(features_list=self.numerical_features,
                                               transformer_type='numerical',
                                               accepts_sparse=self.accepts_sparse)

        categorical_processor = FeatureProcessor(features_list=self.categorical_features,
                                                 transformer_type='categorical',
                                                 accepts_sparse=self.accepts_sparse)

        self.features_union = ('feature_engineering', make_union(numerical_processor,
                                                                 categorical_processor))
        #Third step: missing imputtation
        self.missings_processor =('missings_imputation',SimpleImputer(strategy='constant',fill_value=0))

        return Pipeline([self.preprocessing,self.features_union, self.missings_processor])





