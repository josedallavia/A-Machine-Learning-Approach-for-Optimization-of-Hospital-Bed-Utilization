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
from sklearn.feature_extraction.text import TfidfVectorizer

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
    
    def __init__(self, keys,feature_type='numerical'):
        self.keys = keys
        self.feature_type = feature_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.feature_type == 'categorical':
            return X[self.keys].astype(str)
        elif self.feature_type == 'text':
            if len(self.keys) > 1:
                raise Exception('More than one feature was passed to text processor!')
            return getattr(X, self.keys[0])
        else:
            return X[self.keys]

class CustomScaler(StandardScaler):
    
    def __init__ (self,scale=False, with_mean=True,with_std=True):
        self.scale= scale
        self.with_mean=with_mean
        self.with_std = with_std

        if self.scale:
            super().__init__(with_mean=with_mean,
                         with_std=with_std)
    
    def fit(self,X,y=None):
        self.feature_names = list(X.columns)
        if self.scale:
            return super().fit(X)
        return self

    
    def transform(self,X):
        if self.scale:
            return super().transform(X)
        return X
        
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
    def __init__(self, features_list, feature_type,accepts_sparse=True,scale_numerical=False):
        self.features_list = features_list
        self.feature_type = feature_type
        self.accepts_sparse = accepts_sparse
        self.scale_numerical = scale_numerical


        self.selector = ItemSelector(keys= self.features_list,
                                     feature_type = self.feature_type)

        if self.feature_type == 'categorical':
            self.transformer = CustomEncoder(self.accepts_sparse)
        elif self.feature_type == 'numerical':
            self.transformer = CustomScaler(scale=self.scale_numerical)
        elif self.feature_type == 'text':
            self.transformer = TfidfVectorizer(lowercase=True, ngram_range=(1,4), token_pattern='[^,]+',
                                               min_df=10)
        
        super().__init__([('selector',self.selector),
                          (self.feature_type, self.transformer)])
        
    def get_feature_names(self):
        return self.transformer.get_feature_names()

class CustomPipeline():
    def __init__(self, categorical_features=[],numerical_features=[],text_features=[],
                 accepts_sparse=True, scale_numerical=False):

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.text_features = text_features
        self.accepts_sparse= accepts_sparse
        self.scale_numerical = scale_numerical

    def build_pipeline(self):
        #First step of pipeline: preprocesing
        self.preprocessing = ('preprocessing', FeaturePreProcessor())

        #Second step: feature engineering
        self.features_processors = []

        if len(self.numerical_features) > 0:
            numerical_processor = FeatureProcessor(features_list=self.numerical_features,
                                                    feature_type='numerical',
                                                    accepts_sparse=self.accepts_sparse,
                                                    scale_numerical=self.scale_numerical)
            self.features_processors.append(('numerical_processor',numerical_processor))

        if len(self.categorical_features) > 0:

            categorical_processor = FeatureProcessor(features_list=self.categorical_features,
                                                 feature_type='categorical',
                                                 accepts_sparse=self.accepts_sparse)
            self.features_processors.append(('categorical_processor' ,categorical_processor))

        if len(self.text_features) > 0:
            for text_feature in self.text_features:
                text_processor = FeatureProcessor(features_list=self.text_features,
                                                        feature_type='text',
                                                        accepts_sparse=self.accepts_sparse)

            self.features_processors.append((text_feature,text_processor))


        self.features_union = ('feature_engineering', FeatureUnion(transformer_list=self.features_processors))
        #Third step: missing imputtation
        self.missings_processor =('missings_imputation',SimpleImputer(strategy='constant',fill_value=0))

        return Pipeline([self.preprocessing,self.features_union, self.missings_processor])





