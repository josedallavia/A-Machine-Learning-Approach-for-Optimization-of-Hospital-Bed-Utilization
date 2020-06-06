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
        self.key = keys
        self.is_categorical = is_categorical

    def fit(self, X, y=None):
        return self

    def transform(self, X):     
        return X[self.key].astype(str) if self.is_categorical else X[self.key]
    
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
        super().__init__(handle_unknown='ignore',sparse=accepts_sparse)
        
    def fit(self,X,y=None):
        self.features_headers = list(X.columns)
        return super().fit(X)
        
    def get_feature_names(self):
        return super().get_feature_names(self.features_headers)
    
class FeatureProcessor(Pipeline):
    def __init__(self, features_list, transformer_type,accepts_sparse=True):
        self.features_list = features_list
        self.transformer_dict = {'categorical': CustomEncoder(accepts_sparse),
                                 'numerical': CustomScaler()}
        
        self.selector  = ItemSelector(
            keys=features_list,is_categorical= (transformer_type == 'categorical'))
        
        self.transformer = self.transformer_dict[transformer_type]
        
        super().__init__([('selector',self.selector),
                          (transformer_type, self.transformer)])
        
    def get_feature_names(self):
        return self.transformer.get_feature_names()
    
    
def build_pipeline(categorical_features,
                   numerical_features,
                  accepts_sparse=True):
        
        features_union = make_union(FeatureProcessor(numerical_features ,'numerical'),
                                    FeatureProcessor(categorical_features,'categorical',accepts_sparse))
        
        pipeline = Pipeline(steps=[('preprocessing', FeaturePreProcessor() ),
                                   ('feature_engineering', features_union)
                                  
                                  ])
        print(pipeline)
        
        return pipeline