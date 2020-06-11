import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from thesis_lib.modelling.data import *

hospital_stop_words = ['de','con','en','la','el','para','por','del','izquierda','las',
 'los','izq','izquierda','derecha','otro','otra','otros','otras','paciente','fase', '__','_']

class FeaturePreProcessor():
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X['admission_weekday'] = pd.to_datetime(X['admission_date']).dt.weekday.astype('str')
        X['admission_month'] = pd.to_datetime(X['admission_date']).dt.month.astype('str')
        X['date_weekday'] = pd.to_datetime(X['date']).dt.weekday.astype('str')
        X['date_month'] = pd.to_datetime(X['date']).dt.weekday.astype('str')
        return X

class ItemSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, keys,feature_type='numerical'):
        self.keys = keys
        self.feature_type = feature_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('\t Preprocessing data')
        if self.feature_type == 'categorical':
            return X[self.keys].astype(str)
        elif self.feature_type == 'text':
            if len(self.keys) > 1:
                raise Exception('More than one feature was passed to text processor!')
            return np.array(getattr(X, self.keys[0])).reshape(-1, 1)
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
            print('\t Fitting StandardScaler for numerical features')
            return super().fit(X)
        return self

    
    def transform(self,X):
        if self.scale:
            print('\t Scaling numerical features with StandardScaler')
            return super().transform(X)
        return X
        
    def get_feature_names(self):
        return self.feature_names

class CustomEncoder(OneHotEncoder):
    
    def __init__(self,accepts_sparse=True):
        self.accepts_sparse= accepts_sparse
        super().__init__(handle_unknown='ignore',sparse=accepts_sparse)
        
    def fit(self,X,y=None):
        print('\t Encoding Categorical Features with OneHotEncoding')
        self.features_headers = list(X.columns)
        return super().fit(X)
        
    def get_feature_names(self):
        return super().get_feature_names(self.features_headers)


class CustomImputer(SimpleImputer):
    def __init__(self, strategy='constant', fill_value=''):
        self.strategy = strategy
        self.fill_value = fill_value

        super().__init__(strategy=self.strategy,fill_value=fill_value)

    def fit(self,X,y=None):
        return super().fit(X)

    def fit_transform(self, X, y=None):
        super().fit(X)
        docs = self.transform(X)
        return docs

    def transform(self,X):
        docs = super().transform(X)
        return np.array([str(doc) for doc in docs])

class CustomTfidfVectorizer(TfidfVectorizer):
    def __init__(self,lowercase=True, ngram_range=(1, 4),token_pattern='(?u)\\b\\w\\w+\\b',
                 min_df=10, max_df=0.9,stop_words=hospital_stop_words,prefix=''):
        self.lowercase=lowercase
        self.ngram_range= ngram_range
        self.token_pattern=token_pattern
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words=stop_words
        self.prefix = prefix

        super().__init__(lowercase=self.lowercase, ngram_range=self.ngram_range,
                         token_pattern=self.token_pattern, min_df=self.min_df, max_df=self.max_df,
                         stop_words=self.stop_words)

    def fit(self,X,y=None):
        print('\t Fitting TD-IDF matrix')
        return super().fit(X)

    def transform(self,X):
        print('\t Transforming text features with TF-IDF embeddings')
        return super().transform(X)

    def get_feature_names(self):
        tokens = super().get_feature_names()
        return [self.prefix+'_'+token for token in tokens]


class FeatureProcessor(Pipeline):
    def __init__(self, features_list, feature_type,accepts_sparse=True,scale_numerical=False):
        self.features_list = features_list
        self.feature_type = feature_type
        self.accepts_sparse = accepts_sparse
        self.scale_numerical = scale_numerical

        self.processor_steps = []
        #First step is always selector
        self.processor_steps.append(('selector',
                                     ItemSelector(keys=self.features_list,feature_type=self.feature_type)))

        #next steps depende on the feature type
        if self.feature_type == 'categorical':
            self.transformer = CustomEncoder(self.accepts_sparse)
            self.processor_steps.append((self.feature_type,self.transformer))

        elif self.feature_type == 'numerical':
            self.transformer = CustomScaler(scale=self.scale_numerical)
            self.processor_steps.append((self.feature_type,self.transformer))

        elif self.feature_type == 'text':
            self.processor_steps.append(('missings_imputation',
                                         CustomImputer(strategy='constant', fill_value='null')))
            self.transformer = CustomTfidfVectorizer(lowercase=True, ngram_range=(1,2),
                                                     token_pattern='(?u)\\b\\w\\w+\\b',
                                                     min_df=10,max_df=0.9,prefix=self.features_list[0])
            self.processor_steps.append(('tfidf_text_transformer',self.transformer))

        elif self.feature_type == 'sequence':
            self.processor_steps.append(('missings_imputation',token
                                         CustomImputer(strategy='constant', fill_value='null')))
            self.transformer = CustomTfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                                     token_pattern='[^,]+',
                                                     min_df=10, max_df=0.9,prefix=self.features_list[0])

            self.processor_steps.append(('tfidf_sequence_transformer', self.transformer))
        
        super().__init__(self.processor_steps)

    def fit(self,X,y=None):
        print('\t Fitting processor for {feature_type} features'.format(feature_type=self.feature_type))
        return super().fit(X)

    def transform(self, X):
        print('\t Transforming {feature_type} features'.format(feature_type=self.feature_type))
        return super().transform(X)

    def get_feature_names(self):
        return self.transformer.get_feature_names()

class CustomPipeline():
    def __init__(self, categorical_features=[],numerical_features=[],text_features=[],sequence_features=[],
                 accepts_sparse=True, scale_numerical=False):

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.text_features = text_features
        self.sequence_features = sequence_features
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
                text_processor = FeatureProcessor(features_list=[text_feature],
                                                        feature_type='text',
                                                        accepts_sparse=self.accepts_sparse)

                self.features_processors.append(('text_'+text_feature,text_processor))

        if len(self.sequence_features) > 0:
            for sequence_feature in self.sequence_features:
                sequence_processor = FeatureProcessor(features_list=[sequence_feature],
                                                        feature_type='sequence',
                                                        accepts_sparse=self.accepts_sparse)

                self.features_processors.append(('seq_'+sequence_feature,sequence_processor))


        self.features_union = ('feature_engineering', FeatureUnion(transformer_list=self.features_processors))
        #Third step: missing imputtation
        self.missings_processor =('missings_imputation',SimpleImputer(strategy='constant',fill_value=0))

        return Pipeline([self.preprocessing,self.features_union, self.missings_processor])





