import os
import pandas as pd
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
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


        self.pipeline = CustomPipeline(self.model_params['categorical_features'],
                                       self.model_params['numerical_features'],
                                       self.model_params['accepts_sparse']).build_pipeline()


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
            self.classifier = LGBM_classifier()
            self.classifier.set_params(**params)
        elif self.model_params['classifier'] == 'random_forest':
            self.classifier = RFClassifier()
            self.classifier.set_params(**params)

        print('Training classifier')
        self.classifier.fit(self.X_train, self.y_train)

    def fit_best_classifier(self):
        best_params = self.model_selection.best_params_
        self.fit_classifier(params=best_params)

    def predict(self,X_transf):
        return self.classifier.predict(X_transf)

    def score(self,X_transf, y):
        return self.classifier.score(X_transf, y)

    @property
    def get_feature_importance(self):

        return {'feature_importance': self.classifier.feature_importance_,
                'feature_name': self.model_features}

    def plot_feature_importance(self, n_features=None):

        if not n_features:
            n_features = len(self.model_features)

        importance_df = pd.DataFrame(self.get_feature_importance)
        importance_df.sort_values(by='feature_importance',ascending=True, inplace=True)
        importance_df[:n_features].plot.barh(x=1,y=0, title='Feature Importance',legend=False)


    def get_performance_metrics(self):

        auc_train = self.score(self.X_train, self.y_train)
        auc_val = self.score(self.X_val, self.y_val)

        print('training AUC ROC score: ', auc_train)

        print('validation AUC ROC score: ', auc_val)

        print('relative overfitting: ', abs(auc_train-auc_val)/auc_train)

    def optimize_hyperparams(self, params_dict, n_iter=10, n_folds=5, search_type='random'):

        if self.model_params['classifier'] == 'lgbm':
            tmp_classifier =  LGBM_classifier()
        elif self.model_params['classifier'] == 'random_forest':
            tmp_classifier = RFClassifier()

        if search_type == 'random':
            self.model_selection = RandomizedSearchCV(estimator=tmp_classifier,
                                        param_distributions=params_dict,refit=False,
                                        random_state=2020,n_iter=n_iter,cv=n_folds,
                                        verbose=15)
        elif search_type == 'grid':
            self.model_selection = GridSearchCV(estimator=tmp_classifier,
                                                param_grid=params_dict, refit=False,
                                                cv=n_folds,verbose=15)

        self.model_selection.fit(self.X_train,self.y_train)

        return self.model_selection.cv_results_

    def get_model_selection_results(self):
        results = self.model_selection.cv_results_
        results_df = pd.DataFrame({key: results[key] for key in results
                                if key != 'params'})

        return results_df





